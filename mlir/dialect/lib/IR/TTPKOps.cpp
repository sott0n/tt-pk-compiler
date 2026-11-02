#include "tt_pk/IR/TTPKOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::tt_pk;

LogicalResult CoreSetOp::verify() {
  ArrayRef<int64_t> cores = getCores();
  if (cores.size() % 2 != 0) {
    return emitOpError()
           << "cores must contain an even number of entries (x,y pairs); got "
           << cores.size();
  }
  // NonEmptyCoreSet: a placement with zero cores is never meaningful.
  if (cores.empty()) {
    return emitOpError() << "core_set must list at least one core";
  }
  for (int64_t c : cores) {
    if (c < 0) {
      return emitOpError() << "core coordinates must be non-negative; got "
                           << c;
    }
  }
  // CoresUniqueWithinSet is enforced by ProgramOp::verify because the
  // parent op's verifier fires before the child op's in MLIR, and
  // ProgramOp owns the DenseMap it already builds for CoreSetDisjoint.
  return success();
}

LogicalResult ProgramOp::verify() {
  // CoreSetDisjoint: every coordinate claimed by a tt_pk.core_set op in
  // this program's body may appear at most once. Overlap would mean two
  // placements fight for the same Tensix tile.
  llvm::DenseMap<std::pair<int64_t, int64_t>, Operation *> seen;
  // UniqueKernelName: duplicate names would collide at the Resident
  // runtime's kernel registry.
  llvm::StringMap<Operation *> kernelNames;
  // CoreSetHasConsumer: a tt_pk.core_set with no kernel is dead placement
  // and almost always a search bug.
  llvm::DenseSet<Operation *> consumedSets;

  WalkResult walk = getBody().walk([&](Operation *op) -> WalkResult {
    if (auto cs = dyn_cast<CoreSetOp>(op)) {
      ArrayRef<int64_t> cores = cs.getCores();
      for (size_t i = 0; i + 1 < cores.size(); i += 2) {
        std::pair<int64_t, int64_t> coord{cores[i], cores[i + 1]};
        auto [it, inserted] = seen.try_emplace(coord, cs.getOperation());
        if (!inserted) {
          // Parent's verify fires before the child's, so distinguish
          // within-set dup from cross-set dup here directly.
          if (it->second == cs.getOperation()) {
            cs.emitOpError() << "core (" << coord.first << ","
                             << coord.second
                             << ") listed more than once in the same "
                                "core_set";
          } else {
            cs.emitOpError() << "core (" << coord.first << ","
                             << coord.second
                             << ") already assigned to another core_set in "
                                "this program";
          }
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }
    if (auto k = dyn_cast<KernelOp>(op)) {
      StringRef name = k.getKernelName();
      auto [it, inserted] = kernelNames.try_emplace(name, k.getOperation());
      if (!inserted) {
        k.emitOpError() << "kernel name '" << name
                        << "' already launched by another kernel op in this "
                           "program";
        return WalkResult::interrupt();
      }
      if (Operation *def = k.getCoreSet().getDefiningOp()) {
        consumedSets.insert(def);
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) {
    return failure();
  }

  WalkResult orphans = getBody().walk([&](CoreSetOp cs) -> WalkResult {
    if (!consumedSets.contains(cs.getOperation())) {
      cs.emitOpError() << "core_set is not launched by any tt_pk.kernel op "
                          "in this program";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return orphans.wasInterrupted() ? failure() : success();
}
