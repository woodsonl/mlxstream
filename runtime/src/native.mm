// nanobind layer over metal_core: keeps Python free of Objective-C and the core free of Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include "basalt_cpu.h"
#include "basalt_cpu_experts.h"
#include "basalt_ngram.h"
#include "basalt_ple_table.h"
#include "basalt_row_cache.h"
#include "basalt_direct_file.h"
#include "basalt_pool.h"
#include "basalt_cpu_expf.h"
#include "metal_core.h"

namespace nb = nanobind;
using namespace monolith;

// A host task written in Python: fn(step, point). run() is called with the GIL released, so the call takes it; an
// exception fails the run with its message and stays on the task (take_exception) for the caller to raise. The slot is
// the task's own, so a task serves one runner: Engine wraps each callable in a CallableTask of its own.
struct CallableTask : HostTask {
  nb::callable fn;
  nb::object exception;
  explicit CallableTask(nb::callable f) : fn(std::move(f)) {}
  bool run(const HostCall& call, std::string& error) override {
    nb::gil_scoped_acquire gil;
    if (!fn.is_valid()) { error = "the task's callable has been released"; return false; }
    try {
      fn(call.step, call.point);
      return true;
    } catch (nb::python_error& e) {
      error = e.what();
      exception = nb::borrow(e.value());
      return false;
    }
  }
};

// Test support: a C++ task that throws, at `step`, what a C++ task may throw besides a std::exception: an int, or an
// NSException (`objc`).
struct ThrowingTask : HostTask {
  uint64_t step;
  bool objc;
  ThrowingTask(uint64_t s, bool o) : step(s), objc(o) {}
  bool run(const HostCall& call, std::string&) override {
    if (call.step == step) {
      if (objc) [NSException raise:NSGenericException format:@"thrown at step %llu", (unsigned long long)step];
      throw 42;
    }
    return true;
  }
};

// Garbage-collector support. A task's callable usually reaches the Engine that owns the Runner holding the task, a cycle
// through C++ references the collector cannot see unless these types report and break them.
static int callable_task_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  CallableTask* task = nb::inst_ptr<CallableTask>(self);
  Py_VISIT(task->fn.ptr());
  Py_VISIT(task->exception.ptr());                    // its traceback's frames usually reach the Engine too
  return 0;
}
static int callable_task_clear(PyObject* self) {
  if (!nb::inst_ready(self)) return 0;
  CallableTask* task = nb::inst_ptr<CallableTask>(self);
  nb::callable fn = std::move(task->fn);              // the task holds neither before they die
  nb::object exception = std::move(task->exception);
  return 0;
}
static int runner_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  // A task handed in from Python is a shared_ptr whose deleter holds one reference to the Python object, per control
  // block; a task made in C++ holds none. Visit exactly those references, without making a wrapper.
  std::vector<const void*> blocks;
  for (auto& task : nb::inst_ptr<Runner>(self)->host_tasks()) {
    auto* owner = std::get_deleter<nb::detail::py_deleter>(task);
    if (!owner || std::find(blocks.begin(), blocks.end(), owner) != blocks.end()) continue;
    blocks.push_back(owner);
    Py_VISIT(owner->o);
  }
  return 0;
}
static int runner_clear(PyObject* self) {
  if (!nb::inst_ready(self)) return 0;
  nb::inst_ptr<Runner>(self)->release_host_tasks();
  return 0;
}
static PyType_Slot callable_task_slots[] = {{Py_tp_traverse, (void*)callable_task_traverse}, {Py_tp_clear, (void*)callable_task_clear}, {0, nullptr}};
static PyType_Slot runner_slots[] = {{Py_tp_traverse, (void*)runner_traverse}, {Py_tp_clear, (void*)runner_clear}, {0, nullptr}};

// ---- basalt's CPU kernels (basalt_cpu.h, basalt_cpu_expf.h) ---------------------------------------------------
// Caller-owned numpy arrays, C-contiguous and of the shape each kernel reads: nanobind refuses a wrong dtype, rank
// or layout (the arguments are .noconvert(), so it cannot quietly write into a copy), and these checks refuse a
// short buffer - a kernel here reads and writes raw pointers, so the binding is where "every store is inside its
// binding" is kept. The float32 arrays also keep a bit pattern intact on the way in: a scalar argument would cross
// Python as a double, and float -> double -> float quiets a signalling NaN.
namespace bc = monolith::basalt_cpu;
using F32In1 = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using U8In1 = nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I8Out1 = nb::ndarray<int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I32Out1 = nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F32Out1 = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I8In2 = nb::ndarray<const int8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F32In2 = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F32Out2 = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using U8In2 = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using I32In1 = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I32In2 = nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F32Out3 = nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using U32Out2 = nb::ndarray<uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using U8Out1 = nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static void need(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}

// An elementwise kernel writes out[i] from in[i], so it is well defined when the two are the same buffer and not
// when they merely overlap: a shifted view (x = a[:-1], out = a[1:]) is contiguous and passes every other check,
// and the store to out[i] would change the value read as x[i+1]. Refused rather than copied, because a copy would
// quietly give a different answer than the docstring states where the caller meant the arrays to be distinct.
static void no_partial_overlap(const float* in, const float* out, size_t n, const char* what) {
  need(in == out || in + n <= out || out + n <= in, what);
}

// The same hazard where the two arrays hold different types, which is where it is easiest to write by accident:
// a uint8 view of a float32 output array is contiguous and the right dtype, so every other check in a binding
// passes it, and the write then lands under the read. Compared in bytes, and exact aliasing is refused too,
// because none of these ops is in-place.
static void no_shared_storage(const void* a, size_t a_bytes, const void* b, size_t b_bytes, const char* what) {
  const auto* p = (const unsigned char*) a;
  const auto* q = (const unsigned char*) b;
  need(p + a_bytes <= q || q + b_bytes <= p, what);
}

NB_MODULE(_native, m) {
  m.doc() = "Metal runtime core (v0): device, buffers, runtime-compiled libraries, pipelines, timed dispatches";

  m.def("validation_reports", &validation_reports, "shader-validation reports read from completed command buffers so far");

  nb::class_<DeviceInfo>(m, "DeviceInfo")
      .def_ro("name", &DeviceInfo::name).def_ro("gpu_cores", &DeviceInfo::gpu_cores)
      .def_ro("apple_family", &DeviceInfo::apple_family).def_ro("max_buffer_length", &DeviceInfo::max_buffer_length)
      .def_ro("recommended_working_set", &DeviceInfo::recommended_working_set)
      .def_ro("has_unified_memory", &DeviceInfo::has_unified_memory);

  nb::class_<Device>(m, "Device").def(nb::init<>()).def("info", &Device::info);

  nb::class_<Buffer>(m, "Buffer")
      .def("__init__", [](Buffer* self, const Device& d, size_t nbytes) { new (self) Buffer(d, nbytes); }, nb::arg("device"), nb::arg("nbytes"))
      .def("__init__", [](Buffer* self, const Device& d, nb::bytes data) { new (self) Buffer(d, data.size(), data.c_str()); }, nb::arg("device"), nb::arg("data"))
      .def_static("from_file", [](const Device& d, const std::string& path, uint64_t offset, size_t nbytes) { return new Buffer(d, path, offset, nbytes); },
                  nb::arg("device"), nb::arg("path"), nb::arg("offset"), nb::arg("nbytes"))
      .def_prop_ro("nbytes", &Buffer::nbytes)
      .def_prop_ro("gpu_address", &Buffer::gpu_address)
      .def("read", [](const Buffer& b, size_t offset, size_t n) {
        if (offset + n > b.nbytes()) throw std::out_of_range("Buffer.read out of range");
        return nb::bytes((const char*)b.contents() + offset, n); }, nb::arg("offset") = 0, nb::arg("nbytes") = 0)
      .def("write", [](Buffer& b, nb::bytes data, size_t offset) {
        if (offset + data.size() > b.nbytes()) throw std::out_of_range("Buffer.write out of range");
        std::memcpy((char*)b.contents() + offset, data.c_str(), data.size()); }, nb::arg("data"), nb::arg("offset") = 0)
      .def("fill", [](Buffer& b, uint8_t v) { std::memset(b.contents(), v, b.nbytes()); });

  nb::class_<Library>(m, "Library")
      .def(nb::init<const Device&, const std::string&, const std::map<std::string, std::string>&, uint32_t, bool>(),
           nb::arg("device"), nb::arg("source"), nb::arg("macros") = std::map<std::string, std::string>{},
           nb::arg("language_version") = 0, nb::arg("fast_math") = false);

  nb::class_<Pipeline>(m, "Pipeline")
      .def(nb::init<const Library&, const std::string&, bool>(), nb::arg("library"), nb::arg("function"), nb::arg("support_icb") = false)
      .def_prop_ro("max_threads_per_threadgroup", &Pipeline::max_threads_per_threadgroup)
      .def_prop_ro("thread_execution_width", &Pipeline::thread_execution_width);

  nb::class_<Dispatch>(m, "Dispatch")
      .def(nb::init<>())
      .def("pipeline", [](Dispatch& d, const Pipeline& p) -> Dispatch& { d.pipeline = &p; return d; }, nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
      .def("buffer", [](Dispatch& d, uint32_t index, const Buffer& b, uint64_t offset) -> Dispatch& { d.buffers.push_back({index, &b, offset}); return d; },
           nb::arg("index"), nb::arg("buffer"), nb::arg("offset") = 0, nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
      .def("bytes", [](Dispatch& d, uint32_t index, nb::bytes data) -> Dispatch& {
        d.bytes.push_back({index, std::vector<uint8_t>((const uint8_t*)data.c_str(), (const uint8_t*)data.c_str() + data.size())}); return d; },
           nb::arg("index"), nb::arg("data"), nb::rv_policy::reference_internal)
      .def("threadgroup_memory", [](Dispatch& d, uint32_t index, uint32_t length) -> Dispatch& { d.threadgroup_memory.push_back({index, length}); return d; }, nb::rv_policy::reference_internal)
      .def("grid", [](Dispatch& d, uint32_t x, uint32_t y, uint32_t z) -> Dispatch& { d.grid[0] = x; d.grid[1] = y; d.grid[2] = z; return d; },
           nb::arg("x"), nb::arg("y") = 1, nb::arg("z") = 1, nb::rv_policy::reference_internal)
      .def("threadgroup", [](Dispatch& d, uint32_t x, uint32_t y, uint32_t z) -> Dispatch& { d.threadgroup[0] = x; d.threadgroup[1] = y; d.threadgroup[2] = z; return d; },
           nb::arg("x"), nb::arg("y") = 1, nb::arg("z") = 1, nb::rv_policy::reference_internal)
      .def("barrier", [](Dispatch& d, bool on) -> Dispatch& { d.barrier_before = on; return d; }, nb::arg("on") = true, nb::rv_policy::reference_internal)
      .def("use", [](Dispatch& d, const Buffer& b, bool write) -> Dispatch& { d.uses.push_back({&b, write}); return d; },
           nb::arg("buffer"), nb::arg("write") = false, nb::rv_policy::reference_internal, nb::keep_alive<1, 2>());

  nb::class_<RunResult>(m, "RunResult")
      .def_ro("gpu_ms", &RunResult::gpu_ms).def_ro("wall_ms", &RunResult::wall_ms).def_ro("error", &RunResult::error);

  nb::class_<Queue>(m, "Queue")
      .def(nb::init<const Device&>(), nb::arg("device"))
      .def("run", &Queue::run, nb::arg("dispatches"), nb::arg("concurrent") = false)
      .def("profile", &Queue::profile, nb::arg("dispatches"))
      .def("supports_profiling", &Queue::supports_profiling);

  nb::class_<Icb>(m, "Icb")
      .def(nb::init<const Device&, const std::vector<Dispatch>&>(), nb::arg("device"), nb::arg("ops"), nb::keep_alive<1, 3>())
      .def_prop_ro("count", &Icb::count);

  nb::class_<HostTask>(m, "HostTask");    // implemented in C++ (later tiers) or by CallableTask
  nb::class_<CallableTask, HostTask>(m, "CallableTask", nb::type_slots(callable_task_slots))
      .def(nb::init<nb::callable>(), nb::arg("fn"))
      .def("take_exception", [](CallableTask& t) -> nb::object {
             nb::object e = std::move(t.exception);
             if (!e.is_valid()) return nb::none();
             return e; },
           "the exception the task's last failing call raised (None if none); the task no longer holds it");
  nb::class_<ThrowingTask, HostTask>(m, "_ThrowingTask").def(nb::init<uint64_t, bool>(), nb::arg("step"), nb::arg("objc") = false);

  nb::class_<RunnerStats>(m, "RunnerStats")
      .def_ro("steps_submitted", &RunnerStats::steps_submitted).def_ro("command_buffers", &RunnerStats::command_buffers)
      .def_ro("gpu_ms", &RunnerStats::gpu_ms).def_ro("wall_ms", &RunnerStats::wall_ms).def_ro("host_busy_ms", &RunnerStats::host_busy_ms)
      .def_ro("encode_ms", &RunnerStats::encode_ms).def_ro("commit_ms", &RunnerStats::commit_ms).def_ro("wait_ms", &RunnerStats::wait_ms)
      .def_ro("done", &RunnerStats::done).def_ro("error", &RunnerStats::error)
      .def_ro("failed_task", &RunnerStats::failed_task).def_ro("failed", &RunnerStats::failed).def_ro("handoff_gaps_ms", &RunnerStats::handoff_gaps_ms)
      .def_ro("tokens", &RunnerStats::tokens);

  nb::class_<Runner>(m, "Runner", nb::type_slots(runner_slots))
      .def("__init__", [](Runner* self, const Device& d, const Icb& icb, const std::vector<Dispatch>& ops, std::vector<const Buffer*> resources,
                          const Buffer& state, uint32_t done_offset, uint32_t ring_head_offset, uint32_t ring_tail_offset, const Buffer& ring, uint32_t ring_capacity,
                          std::vector<const Buffer*> read_only_resources) {
             new (self) Runner(d, icb, ops, resources, state, done_offset, ring_head_offset, ring_tail_offset, ring, ring_capacity, read_only_resources); },
           nb::arg("device"), nb::arg("icb"), nb::arg("ops"), nb::arg("resources"), nb::arg("step_state"), nb::arg("done_offset"),
           nb::arg("ring_head_offset"), nb::arg("ring_tail_offset"), nb::arg("ring"), nb::arg("ring_capacity"),
           nb::arg("read_only_resources") = std::vector<const Buffer*>{},
           nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(), nb::keep_alive<1, 6>(), nb::keep_alive<1, 10>())
      .def("run", &Runner::run, nb::arg("max_steps"), nb::arg("steps_per_cb") = 8, nb::arg("in_flight") = 3, nb::arg("reencode") = false, nb::arg("max_tokens") = 0,
           nb::call_guard<nb::gil_scoped_release>())
      .def("set_host_points", [](Runner& r, const std::vector<std::tuple<uint32_t, uint32_t, std::shared_ptr<HostTask>>>& points,
                                 uint32_t error_offset, uint32_t timeout_ms, uint32_t spin_us) {
             std::vector<HostPoint> pts;
             for (auto& [publish, wait, task] : points) pts.push_back({publish, wait, task});
             r.set_host_points(std::move(pts), error_offset, timeout_ms, spin_us); },
           nb::arg("points"), nb::arg("error_offset"), nb::arg("timeout_ms") = 20000, nb::arg("spin_us") = 20000)
      .def("host_tasks", &Runner::host_tasks, "the installed tasks, one per host point in program order")
      .def("drain", &Runner::drain);

  m.def("basalt_act_quant_q8_1", [](F32In1 x, I8Out1 q, F32Out1 scale, I32Out1 sum, F32Out1 hx) {
    const size_t len = x.shape(0);             // checked at full width: a narrowing cast first would accept a
    need(len > 0 && len % bc::QKA == 0 && len <= (size_t)bc::H,   // length whose low 32 bits look valid
         "basalt_act_quant_q8_1: len(x) must be a positive multiple of 32, at most 2560");
    const int n = (int)len;
    need(q.shape(0) == (size_t)bc::H, "basalt_act_quant_q8_1: q must hold H = 2560 int8 codes");
    need(scale.shape(0) == (size_t)bc::MAXC && sum.shape(0) == (size_t)bc::MAXC && hx.shape(0) == (size_t)bc::MAXC,
         "basalt_act_quant_q8_1: scale, sum and hx must each hold MAXC = 80 elements");
    bc::ActQ a;
    nb::gil_scoped_release unlock;
    bc::act_quant_q8_1(x.data(), n, a);
    std::memcpy(q.data(), a.q, (size_t)n);                                      // only what the kernel wrote
    std::memcpy(scale.data(), a.scale, (size_t)a.nchunks * sizeof(float));
    std::memcpy(sum.data(), a.sum, (size_t)a.nchunks * sizeof(int32_t));
    std::memcpy(hx.data(), a.hx, (size_t)a.nchunks * sizeof(float));
    return a.nchunks;
  }, nb::arg("x").noconvert(), nb::arg("q").noconvert(), nb::arg("scale").noconvert(), nb::arg("sum").noconvert(),
     nb::arg("hx").noconvert(),
     "basalt's act_quant_q8_1 on ARM NEON: writes q[:n] and scale/sum/hx[:n // 32], and returns nchunks");

  m.def("basalt_q2_0_gguf_rows_multi", [](U8In1 w, size_t row_bytes, int nblocks, I8In2 q, F32In2 scale, F32In2 hx,
                                          F32Out2 out, int r0, int r1) {
    const int nt = (int)q.shape(0);
    const size_t nrows = out.shape(1);        // full width: the row stride below, and a narrowing cast here would
                                              // accept a shape whose low 32 bits look small and stride past it
    need(nt >= 1 && nt <= bc::MAXT, "basalt_q2_0_gguf_rows_multi: nt must be 1..8");
    need(scale.shape(0) == (size_t)nt && hx.shape(0) == (size_t)nt && out.shape(0) == (size_t)nt,
         "basalt_q2_0_gguf_rows_multi: q, scale, hx and out must agree on nt");
    need(q.shape(1) == (size_t)bc::H, "basalt_q2_0_gguf_rows_multi: q rows must hold H = 2560 int8 codes");
    need(scale.shape(1) == (size_t)bc::MAXC && hx.shape(1) == (size_t)bc::MAXC,
         "basalt_q2_0_gguf_rows_multi: scale and hx rows must hold MAXC = 80 elements");
    // Every bound divides rather than multiplies: the products overflow, and a wrapped product passes the check it
    // is meant to fail (nblocks = 2^25 makes QK * nblocks reach INT_MIN, row_bytes * nrows reaches 2^64 and wraps
    // to 0), which the kernel then reads past the ActQ arrays and past w.
    need(nblocks >= 1 && nblocks <= bc::H / bc::QK, "basalt_q2_0_gguf_rows_multi: nblocks must be 1..40");
    need(row_bytes >= (size_t)bc::BB * (size_t)nblocks, "basalt_q2_0_gguf_rows_multi: row_bytes is below 18 * nblocks");
    need(r0 >= 0 && r1 >= r0 && (size_t)r1 <= nrows, "basalt_q2_0_gguf_rows_multi: need 0 <= r0 <= r1 <= out.shape[1]");
    need(nrows == 0 || row_bytes <= w.size() / nrows, "basalt_q2_0_gguf_rows_multi: w is shorter than row_bytes * nrows");
    std::array<bc::ActQ, bc::MAXT> act{};
    const bc::ActQ* ap[bc::MAXT];
    float* op[bc::MAXT];
    for (int t = 0; t < nt; ++t) {
      std::memcpy(act[t].q, q.data() + (size_t)t * bc::H, bc::H);
      std::memcpy(act[t].scale, scale.data() + (size_t)t * bc::MAXC, bc::MAXC * sizeof(float));
      std::memcpy(act[t].hx, hx.data() + (size_t)t * bc::MAXC, bc::MAXC * sizeof(float));
      ap[t] = &act[t];                                                          // sum and nchunks stay zero: unread
      op[t] = out.data() + (size_t)t * nrows;   // nrows is size_t: the stride is the row's real width
    }
    nb::gil_scoped_release unlock;
    bc::q2_0_gguf_rows_multi(w.data(), row_bytes, nblocks, ap, nt, op, r0, r1);
  }, nb::arg("w").noconvert(), nb::arg("row_bytes"), nb::arg("nblocks"), nb::arg("q").noconvert(),
     nb::arg("scale").noconvert(), nb::arg("hx").noconvert(), nb::arg("out").noconvert(), nb::arg("r0"), nb::arg("r1"),
     "basalt's q2_0_gguf_rows_multi on ARM NEON: writes out[t][r] for r0 <= r < r1 and nothing else");

  m.def("basalt_expf", [](F32In1 x, F32Out1 out) {
    need(out.shape(0) == x.shape(0), "basalt_expf: out must hold as many floats as x");
    const size_t n = x.shape(0);
    const float* xp = x.data();
    float* op = out.data();
    no_partial_overlap(xp, op, n, "basalt_expf: out may be x itself but must not partly overlap it");
    nb::gil_scoped_release unlock;
    for (size_t i = 0; i < n; ++i) op[i] = bc::expf_glibc(xp[i]);
  }, nb::arg("x").noconvert(), nb::arg("out").noconvert(),
     "glibc's expf, element by element: out[i] = expf_glibc(x[i])");

  m.def("basalt_swiglu", [](F32In1 g, F32In1 u, F32Out1 out) {
    need(u.shape(0) == g.shape(0) && out.shape(0) == g.shape(0),
         "basalt_swiglu: g, u and out must hold the same number of floats");
    const size_t n = g.shape(0);
    const float* gp = g.data();
    const float* up = u.data();
    float* op = out.data();
    no_partial_overlap(gp, op, n, "basalt_swiglu: out may be g itself but must not partly overlap it");
    no_partial_overlap(up, op, n, "basalt_swiglu: out may be u itself but must not partly overlap it");
    nb::gil_scoped_release unlock;
    for (size_t i = 0; i < n; ++i) op[i] = bc::swiglu(gp[i], up[i]);
  }, nb::arg("g").noconvert(), nb::arg("u").noconvert(), nb::arg("out").noconvert(),
     "basalt's CPU SwiGLU (pool.cpp:434): out[i] = (g[i] / (1.f + expf_glibc(-g[i]))) * u[i]");

  // The acceptance oracle sweeps all 2^32 float32 arguments, which no Python loop can hash in reasonable time and
  // which would be 16 GB as an array. The digest runs here instead, over one range of bit patterns per call: the
  // FNV-1a of the result bits in increasing argument order, the rule tests/kernels/fixtures/expf_digests.txt
  // records glibc's own output under.
  // ---- the CPU expert pool (basalt_pool.h) ----------------------------------------------------------------
  // Driven from the tests with a native body, never a Python callable: a body that took the GIL would serialise
  // every worker and the synchronisation this pool exists to get right would never be exercised.
  nb::class_<bc::ExpertPool>(m, "ExpertPool")
      .def("__init__", [](bc::ExpertPool* self, int n_workers, bool host_works, double spin_before_sleep_ms) {
             // Checked before the conversion: a NaN, an infinity, or a value past the integer's range makes the
             // float-to-integer conversion undefined rather than merely wrong.
             need(std::isfinite(spin_before_sleep_ms) && spin_before_sleep_ms >= 0.0 &&
                      spin_before_sleep_ms <= 24.0 * 60.0 * 60.0 * 1000.0,
                  "ExpertPool: spin_before_sleep_ms must be finite and 0 to 86,400,000");
             new (self) bc::ExpertPool(n_workers, host_works,
                                       std::chrono::microseconds((long long) (spin_before_sleep_ms * 1000.0)));
           },
           nb::arg("n_workers") = 0, nb::arg("host_works") = true, nb::arg("spin_before_sleep_ms") = 20.0)
      .def_prop_ro("workers", &bc::ExpertPool::workers)
      .def_prop_ro("host_works", &bc::ExpertPool::host_works)
      .def_prop_ro("poisoned", &bc::ExpertPool::poisoned)
      .def("claim_owner", [](bc::ExpertPool& p, uintptr_t owner) { return p.claim_owner((const void*) owner); })
      .def("release_owner", [](bc::ExpertPool& p, uintptr_t owner) { p.release_owner((const void*) owner); })
      // One phase whose job j does out[j] += 1 and records the worker that ran it in who[j]. Every job running
      // exactly once per phase is the property the epoch-tagged claim exists to give.
      .def("count_phase", [](bc::ExpertPool& p, I32Out1 out, I32Out1 who) {
             need(who.shape(0) == out.shape(0), "count_phase: who must hold as many ints as out");
             need(out.shape(0) <= (size_t) 0xffff, "count_phase: a pool batch holds 0 to 65535 jobs");
             const int n = (int) out.shape(0);                     // bounded above, so the narrowing is exact
             int32_t* op = out.data();
             int32_t* wp = who.data();
             // Two different arrays, not merely two of the same length: sharing storage would let the worker-id
             // write replace the count, and a shifted view would let two jobs write one element concurrently.
             need(op + n <= wp || wp + n <= op, "count_phase: out and who must not share storage");
             std::string err;
             bc::PhaseTimes t;
             bool ok;
             {
               nb::gil_scoped_release unlock;
               ok = p.run_phase(n, [op, wp](int job, int id) { op[job] += 1; wp[job] = id; }, err, &t);
             }
             need(ok, err.c_str());
             return nb::make_tuple(t.wait_parked_before_ms, t.work_ms, t.wait_parked_after_ms);
           }, nb::arg("out").noconvert(), nb::arg("who").noconvert(),
           "run one phase of out.size() jobs; job j does out[j] += 1 and writes its worker id to who[j]")
      // One layer's CPU experts, through the two phases. The activations are a shared pool, one row per token,
      // as basalt shares one ActQ per token per layer across every job of the batch; job j takes the tokens
      // job_tok[j][0 .. job_nt[j]) out of it and writes out[j][t].
      .def("run_experts", [](bc::ExpertPool& p, U8In2 blobs, size_t gu_row, size_t d_row, size_t up_off,
                             size_t down_off, I8In2 q, F32In2 scale, F32In2 hx, I32In1 job_nt, I32In2 job_tok,
                             F32Out3 out) {
             const int njobs = (int) blobs.shape(0), ntok = (int) q.shape(0);
             const size_t blob_bytes = blobs.shape(1);
             need(njobs >= 1, "run_experts: needs at least one job");
             need(job_nt.shape(0) == (size_t) njobs && job_tok.shape(0) == (size_t) njobs,
                  "run_experts: job_nt and job_tok must have one row per blob");
             need(job_tok.shape(1) == (size_t) bc::MAXT, "run_experts: job_tok rows hold MAXT = 8 entries");
             need(out.shape(0) == (size_t) njobs && out.shape(1) == (size_t) bc::MAXT &&
                      out.shape(2) == (size_t) bc::H,
                  "run_experts: out must be (njobs, 8, 2560)");
             need(q.shape(1) == (size_t) bc::H && scale.shape(1) == (size_t) bc::MAXC &&
                      hx.shape(1) == (size_t) bc::MAXC && scale.shape(0) == (size_t) ntok &&
                      hx.shape(0) == (size_t) ntok,
                  "run_experts: the activation pool must be (ntok, 2560) and (ntok, 80)");
             // Every bound divides rather than multiplies, as the row kernel's own binding does.
             need(gu_row >= (size_t) bc::BB * (bc::H / bc::QK), "run_experts: gu_row is below 18 * 40");
             need(d_row >= (size_t) bc::BB * (bc::FF / bc::QK), "run_experts: d_row is below 18 * 10");
             need(up_off <= blob_bytes && down_off <= blob_bytes, "run_experts: up_off and down_off run past a blob");
             need(bc::FF == 0 || gu_row <= (blob_bytes - up_off) / bc::FF,
                  "run_experts: the up rows run past the blob");
             need(bc::H == 0 || d_row <= (blob_bytes - down_off) / bc::H,
                  "run_experts: the down rows run past the blob");

             // Before the buffers exist, so a retry against a pool that stalled earlier cannot allocate and then
             // leak them: every phase on a poisoned pool fails, and the failure path below retains what it was
             // given.
             need(!p.poisoned(), "run_experts: the CPU expert pool stalled earlier and cannot be reused");
             std::vector<bc::ActQ> acts((size_t) ntok);
             for (int t = 0; t < ntok; ++t) {
               std::memcpy(acts[(size_t) t].q, q.data() + (size_t) t * bc::H, bc::H);
               std::memcpy(acts[(size_t) t].scale, scale.data() + (size_t) t * bc::MAXC, bc::MAXC * sizeof(float));
               std::memcpy(acts[(size_t) t].hx, hx.data() + (size_t) t * bc::MAXC, bc::MAXC * sizeof(float));
               acts[(size_t) t].nchunks = bc::H / bc::QKA;
             }
             std::vector<bc::ExpertJobMulti> jobs((size_t) njobs);
             for (int j = 0; j < njobs; ++j) {
               const int nt = job_nt.data()[j];
               need(nt >= 1 && nt <= bc::MAXT, "run_experts: every job carries 1 to 8 tokens");
               jobs[(size_t) j].blob = blobs.data() + (size_t) j * blob_bytes;
               jobs[(size_t) j].nt = nt;
               for (int t = 0; t < nt; ++t) {
                 const int tok = job_tok.data()[(size_t) j * bc::MAXT + t];
                 need(tok >= 0 && tok < ntok, "run_experts: a job names a token outside the activation pool");
                 jobs[(size_t) j].act[t] = &acts[(size_t) tok];
                 jobs[(size_t) j].out[t] = out.data() + ((size_t) j * bc::MAXT + t) * bc::H;
               }
             }
             bc::NativeFmt fmt;
             fmt.gu_row = gu_row;
             fmt.d_row = d_row;
             fmt.up_off = up_off;
             fmt.down_off = down_off;
             std::vector<bc::SplitBufMulti> scratch((size_t) std::min(njobs, bc::kMaxSplitMulti));
             std::string err;
             bc::PhaseTimes gu, dn;
             bool ok;
             {
               nb::gil_scoped_release unlock;
               ok = bc::run_split_multi_native(p, fmt, jobs.data(), njobs, scratch.data(), err, &gu, &dn);
             }
             if (!ok && p.poisoned()) {
               // A stalled phase leaves a worker that may still be reading these: the jobs name the blobs and the
               // activations, and the scratch holds the intermediate it writes. There is no way to interrupt it,
               // so they are leaked on purpose for the process's life, the same bargain PoolTask::abandon makes.
               // `out` is the caller's array and is not ours to leak; the suite keeps a poisoned pool and its
               // arrays alive, as it does for the pool's own stall case.
               (void) new std::vector<bc::ActQ>(std::move(acts));
               (void) new std::vector<bc::ExpertJobMulti>(std::move(jobs));
               (void) new std::vector<bc::SplitBufMulti>(std::move(scratch));
             }
             need(ok, err.c_str());
             return nb::make_tuple(gu.work_ms, dn.work_ms);
           }, nb::arg("blobs").noconvert(), nb::arg("gu_row"), nb::arg("d_row"), nb::arg("up_off"),
           nb::arg("down_off"), nb::arg("q").noconvert(), nb::arg("scale").noconvert(), nb::arg("hx").noconvert(),
           nb::arg("job_nt").noconvert(), nb::arg("job_tok").noconvert(), nb::arg("out").noconvert(),
           "one layer's CPU experts through the pool's two phases")
      .def("stall_phase", [](bc::ExpertPool& p, double hold_s) {
             need(std::isfinite(hold_s) && hold_s >= 0.0 && hold_s <= 24.0 * 60.0 * 60.0,
                  "stall_phase: hold_s must be finite and 0 to 86,400");
             // One job that outlives kStall, so the stall path and the poisoning can be tested. A poisoned pool's
             // worker is still inside this body, so neither the body nor the pool may be freed. The body is on the
             // heap and is leaked on the poisoning path, and the pool is held through a non-owning reference the
             // binding cannot free, but the Python object CAN still be collected, so the test keeps its own
             // reference. That is the same bargain PoolTask::abandon makes.
             std::string err;
             bool ok;
             {
               nb::gil_scoped_release unlock;
               auto* body = new std::function<void(int, int)>([hold_s](int, int) {
                 std::this_thread::sleep_for(std::chrono::duration<double>(hold_s));
               });
               ok = p.run_phase(1, *body, err, nullptr);
               if (ok || !p.poisoned()) delete body;               // nothing can still be running it
             }
             return nb::make_tuple(ok, err);
           }, nb::arg("hold_s"), "run one job that sleeps hold_s seconds, so a wait can be made to stall");

  // ---- the PLE n-gram hash (basalt_ngram.h) ---------------------------------------------------------------
  // Sixteen row indices per token. Host-side by construction, so this is the whole kernel: the GPU only gathers
  // the rows it names. The bounds divide rather than multiply, as the other array bindings here do.
  m.def("basalt_ngram_rows", [](I32In1 tokens, I32In2 prev, U32Out2 out) {
          const size_t n = tokens.shape(0);
          need(prev.shape(0) == n && out.shape(0) == n,
               "basalt_ngram_rows: tokens, prev and out must have one row per token");
          need(prev.shape(1) == (size_t) (bc::NGRAM_SIZE - 1), "basalt_ngram_rows: prev rows hold 2 predecessors");
          need(out.shape(1) == (size_t) bc::PLE_N_HEADS, "basalt_ngram_rows: out rows hold 16 head indices");
          // A decode window is tens of tokens; this bound exists so the row count cannot approach the
          // arithmetic's limits, not because a larger batch is meaningful.
          need(n <= (size_t) (1 << 20), "basalt_ngram_rows: at most 1,048,576 tokens");
          // Separate arrays, not views of one: out is written while BOTH inputs are read, and a view of out's
          // own storage would let the write land under the read. tokens aliases just as easily as prev, because
          // a contiguous int32 view of the uint32 out array passes every other check here, and the 16 words
          // written for token 0 sit exactly where tokens 1 through 15 are read from. The same storage check the
          // expert bindings make, against each input in turn.
          const size_t ob = n * bc::PLE_N_HEADS * sizeof(uint32_t);
          no_shared_storage(tokens.data(), n * sizeof(int32_t), out.data(), ob,
                            "basalt_ngram_rows: tokens and out must not share storage");
          no_shared_storage(prev.data(), n * (bc::NGRAM_SIZE - 1) * sizeof(int32_t), out.data(), ob,
                            "basalt_ngram_rows: prev and out must not share storage");
          bc::PleConsts c;
          {
            nb::gil_scoped_release unlock;
            bc::ngram_rows(tokens.data(), prev.data(), (int) n, c, out.data());
          }
        }, nb::arg("tokens").noconvert(), nb::arg("prev").noconvert(), nb::arg("out").noconvert(),
        "sixteen PLE table row indices per token, from the token and its two predecessors (oldest first)");

  // The hash constants, so a test can check the offsets are the running sum of the vocabulary sizes. basalt keeps
  // both arrays in the GGUF metadata and writes both out rather than deriving one, so that a mismatch between
  // them surfaces instead of hiding; this exposes them for exactly that check.
  m.def("basalt_ngram_consts", []() {
          bc::PleConsts c;
          std::vector<uint64_t> mult(c.mult, c.mult + bc::NGRAM_SIZE);
          std::vector<uint64_t> vocab(c.vocab, c.vocab + bc::PLE_N_HEADS);
          std::vector<uint64_t> offset(c.offset, c.offset + bc::PLE_N_HEADS);
          return nb::make_tuple(mult, vocab, offset);
        }, "the n-gram hash constants as (mult, vocab, offset)");

  // ---- the PLE table decode (basalt_ple_table.h) ----------------------------------------------------------
  // Every one of the 65,536 fp16 bit patterns widened exactly, so a test can hold the whole domain at once
  // rather than sampling it. Subnormals are the half that matters: a hardware convert with FZ16 set flushes
  // them, and 0x0001 would come back 0.0 instead of 2^-24.
  m.def("basalt_f16_widen_all", [](F32Out1 out) {
          need(out.shape(0) == 65536, "basalt_f16_widen_all: out must hold 65536 floats");
          float* o = out.data();
          nb::gil_scoped_release unlock;
          for (uint32_t h = 0; h < 65536u; ++h) o[h] = bc::f32_from_f16((uint16_t) h);
        }, nb::arg("out").noconvert(), "widen every fp16 bit pattern to f32, in bit-pattern order");

  m.def("basalt_iq4nl_dequant_row", [](U8In1 row, F32Out1 out) {
          need(row.shape(0) == (size_t) bc::PLE_ROW_BYTES, "basalt_iq4nl_dequant_row: a row is 90 bytes");
          need(out.shape(0) == (size_t) bc::PLE_HEAD_DIM, "basalt_iq4nl_dequant_row: out holds 160 floats");
          no_shared_storage(row.data(), (size_t) bc::PLE_ROW_BYTES, out.data(),
                            (size_t) bc::PLE_HEAD_DIM * sizeof(float),
                            "basalt_iq4nl_dequant_row: row and out must not share storage");
          nb::gil_scoped_release unlock;
          bc::iq4nl_dequant_row(row.data(), out.data());
        }, nb::arg("row").noconvert(), nb::arg("out").noconvert(), "one 90-byte IQ4_NL row to 160 floats");

  m.def("basalt_ple_rows_to_embd", [](U8In1 rows16, F32Out1 out) {
          need(rows16.shape(0) == (size_t) (bc::PLE_N_HEADS * bc::PLE_ROW_BYTES),
               "basalt_ple_rows_to_embd: rows16 must be 16 rows of 90 bytes");
          need(out.shape(0) == (size_t) bc::NG_N_EMBD, "basalt_ple_rows_to_embd: out holds 2560 floats");
          no_shared_storage(rows16.data(), (size_t) (bc::PLE_N_HEADS * bc::PLE_ROW_BYTES), out.data(),
                            (size_t) bc::NG_N_EMBD * sizeof(float),
                            "basalt_ple_rows_to_embd: rows16 and out must not share storage");
          nb::gil_scoped_release unlock;
          bc::ple_rows_to_embd(rows16.data(), out.data());
        }, nb::arg("rows16").noconvert(), nb::arg("out").noconvert(),
        "sixteen decoded rows flattened head-slowest into one 2560-wide embedding");

  m.def("basalt_iq4nl_codebook", []() {
          return std::vector<int>(bc::kIq4Nl, bc::kIq4Nl + 16);
        }, "IQ4_NL's codebook at the ported commit");

  // ---- the PLE row cache (basalt_row_cache.h) ------------------------------------------------------------
  // Bound as a class because its behaviour IS its state: the structural properties worth testing (round robin
  // rather than least-recently-used, the modulo set index, the fill counter) are only visible across a sequence
  // of inserts and lookups.
  nb::class_<bc::RowCache>(m, "RowCache")
      .def("__init__", [](bc::RowCache* self, uint64_t rows, uint32_t row_bytes) {
             need(row_bytes >= 1 && row_bytes <= 4096, "RowCache: row_bytes must be 1 to 4096");
             need(rows <= (uint64_t) 1 << 24, "RowCache: rows must be at most 16,777,216");
             new (self) bc::RowCache();
             self->init(rows, row_bytes);
           }, nb::arg("rows"), nb::arg("row_bytes"))
      .def_prop_ro("capacity", &bc::RowCache::capacity)
      .def_prop_ro("size", &bc::RowCache::size)
      .def_prop_ro("sets", &bc::RowCache::sets)
      .def_static("mix", &bc::RowCache::mix, nb::arg("row"))
      .def("set_of", &bc::RowCache::set_of, nb::arg("row"))
      .def("insert", [](bc::RowCache& c, uint32_t row, U8In1 bytes) {
             // insert copies row_bytes from this pointer, so a shorter array would be read past its end.
             need(bytes.shape(0) == (size_t) c.row_bytes(), "RowCache.insert: bytes must be row_bytes long");
             c.insert(row, bytes.data());
           }, nb::arg("row"), nb::arg("bytes").noconvert())
      .def("get", [](bc::RowCache& c, uint32_t row, U8Out1 out) {
             // A slot holds exactly row_bytes, so a longer `out` would read past it, and past the whole data
             // array on the last slot.
             need(out.shape(0) == (size_t) c.row_bytes(), "RowCache.get: out must be row_bytes long");
             const uint8_t* p = c.find(row);
             if (p == nullptr) return false;
             std::memcpy(out.data(), p, out.shape(0));
             return true;
           }, nb::arg("row"), nb::arg("out").noconvert(),
           "copy the row's bytes into out and return true, or return false when the row is absent");

  // ---- the unbuffered reader (basalt_direct_file.h) -------------------------------------------------------
  // Bound so a test can hold the properties that matter and that no amount of reading the code establishes: the
  // alignment refusal, the short read at the end of the file, and that completions are keyed by tag rather than
  // ordered by submission.
  nb::class_<bc::DirectFile>(m, "DirectFile")
      .def(nb::init<>())
      // Both take the drive lock, for the same reason the reads do. `close` sets `stop_`, joins every worker,
      // and THEN clears `done_`, so a close landing while a driver waits wipes the completion that driver is
      // blocked on and leaves no worker to post another: `wait` with a negative timeout then never returns.
      // `open` calls `close` first, so it carries the same hazard. The GIL is released before the lock is
      // taken, never after: a driver holds the lock while it waits, and needs the GIL back to return.
      .def("open", [](bc::DirectFile& f, const std::string& path, int threads) {
             std::string err;
             nb::gil_scoped_release unlock;
             std::lock_guard<std::mutex> drive(f.drive_mutex());
             if (!f.open(path, err, threads)) throw std::invalid_argument(err);
           }, nb::arg("path"), nb::arg("threads") = bc::DIRECT_IO_THREADS)
      .def("close", [](bc::DirectFile& f) {
             nb::gil_scoped_release unlock;
             std::lock_guard<std::mutex> drive(f.drive_mutex());
             f.close();
           })
      .def_prop_ro("size", &bc::DirectFile::size)
      .def_prop_ro("threads", &bc::DirectFile::threads)
      .def_prop_ro("is_open", &bc::DirectFile::is_open)
      // Reads `length` bytes at `offset` into a fresh aligned buffer and returns them, so the Python side never
      // has to produce a 4096-aligned array of its own. One read per call, waited on here.
      .def("read_at", [](bc::DirectFile& f, uint64_t offset, uint32_t length, uint64_t tag) {
             need(length > 0 && length % bc::DIRECT_ALIGNMENT == 0,
                  "read_at: length must be a nonzero multiple of 4096");
             // The wake tag is the reader's own. A read submitted under it matches the completion `wake` posts,
             // so the wait returns before the read has run and the buffer is freed while a pool thread is still
             // writing into it: a write to freed memory, reachable from one thread with a wake already queued.
             need(tag != bc::DIRECT_WAKE_TAG, "read_at: tag 0xFFFFFFFFFFFFFFFF is the wake tag, not a read's");
             std::vector<uint8_t> bytes;
             uint32_t got = 0;
             bool ok = false;
             {
               nb::gil_scoped_release unlock;
               // Held across the whole submit-and-collect cycle: the completion queue is shared, so a second
               // driver would take this read's completion and then wait forever for one already consumed.
               std::lock_guard<std::mutex> drive(f.drive_mutex());
               void* buf = bc::DirectFile::alloc_aligned(length);
               if (buf == nullptr) throw std::bad_alloc();
               std::string err;
               if (!f.submit(offset, buf, length, tag, err)) {
                 bc::DirectFile::free_aligned(buf);     // nothing was queued, so nothing can still write to it
                 throw std::invalid_argument(err);
               }
               // Keyed by tag, never by arrival: a wake posted earlier, or any other completion still
               // queued, arrives first and is not this read. Taking the front completion is the exact mistake
               // this class's contract warns about.
               bc::Completion c{};
               for (bool mine = false; !mine;) {
                 bc::Completion batch[64];
                 const int n = f.wait(batch, 64, -1);
                 for (int k = 0; k < n; ++k) {
                   if (batch[k].tag != tag) continue;       // a wake, or another caller's read
                   c = batch[k];
                   mine = true;
                 }
               }
               ok = c.ok;
               got = c.bytes;
               bytes.assign((const uint8_t*) buf, (const uint8_t*) buf + (ok ? got : 0u));
               bc::DirectFile::free_aligned(buf);
             }
             return nb::make_tuple(ok, got, nb::bytes((const char*) bytes.data(), bytes.size()));
           }, nb::arg("offset"), nb::arg("length"), nb::arg("tag") = 0,
           "one aligned read; returns (ok, bytes_read, data)")
      // Submits every request before waiting, so a test can see that completions come back keyed by tag rather
      // than in submission order.
      .def("read_many", [](bc::DirectFile& f, std::vector<uint64_t> offsets, uint32_t length) {
             need(length > 0 && length % bc::DIRECT_ALIGNMENT == 0,
                  "read_many: length must be a nonzero multiple of 4096");
             need(!offsets.empty() && offsets.size() <= 4096, "read_many: 1 to 4096 offsets");
             // Every offset checked BEFORE anything is submitted, so a bad one later in the list cannot
             // leave earlier reads queued against buffers this call is about to free.
             for (uint64_t off : offsets)
               need(off % bc::DIRECT_ALIGNMENT == 0, "read_many: every offset must be a multiple of 4096");

             std::vector<std::pair<uint64_t, uint32_t>> order;
             {
               nb::gil_scoped_release unlock;
               std::lock_guard<std::mutex> drive(f.drive_mutex());
               std::vector<void*> bufs(offsets.size(), nullptr);
               size_t submitted = 0;
               std::string err;
               bool failed = false, oom = false;
               for (size_t i = 0; i < offsets.size() && !failed; ++i) {
                 bufs[i] = bc::DirectFile::alloc_aligned(length);
                 if (bufs[i] == nullptr) { failed = oom = true; break; }
                 if (!f.submit(offsets[i], bufs[i], length, (uint64_t) i, err)) { failed = true; break; }
                 ++submitted;
               }
               // Collect every read that WAS submitted before any buffer is freed, whether or not the rest
               // failed: a worker thread still holds a pointer into one of these and writes through it, so
               // freeing early is a write to freed memory rather than a leak.
               size_t seen = 0;
               while (seen < submitted) {
                 bc::Completion got[64];
                 const int n = f.wait(got, 64, -1);
                 for (int k = 0; k < n; ++k) {
                   if (got[k].tag >= submitted) continue;        // a wake, or a tag this call never issued
                   if (!failed) order.emplace_back(got[k].tag, got[k].bytes);
                   ++seen;
                 }
               }
               for (void* b : bufs) bc::DirectFile::free_aligned(b);
               if (oom) throw std::bad_alloc();
               if (failed) throw std::invalid_argument(err);
             }
             return order;
           }, nb::arg("offsets"), nb::arg("length"),
           "submit every offset, then collect; returns (tag, bytes) in completion order")
      .def("wake", &bc::DirectFile::wake);

  m.def("basalt_expf_digest", [](uint32_t first_bits, uint32_t count) {
    // Refused rather than wrapped: a range that runs off the end of the bit patterns would silently digest a
    // different set of arguments than the caller named.
    need(uint64_t(first_bits) + count <= (uint64_t(1) << 32),
         "basalt_expf_digest: first_bits + count must not run past 2^32");
    uint64_t h = 0xCBF29CE484222325ull;
    nb::gil_scoped_release unlock;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t bits = first_bits + i;
      float x;
      std::memcpy(&x, &bits, sizeof x);
      const float y = bc::expf_glibc(x);
      uint32_t yb;
      std::memcpy(&yb, &y, sizeof yb);
      h = (h ^ yb) * 0x100000001B3ull;
    }
    return h;
  }, nb::arg("first_bits"), nb::arg("count"),
     "FNV-1a over the result bits of expf_glibc at `count` consecutive float32 bit patterns from `first_bits`");
}
