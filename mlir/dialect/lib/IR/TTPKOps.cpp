#include "tt_pk/IR/TTPKOps.h"

#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::tt_pk;

LogicalResult CoreSetOp::verify() {
  ArrayRef<int64_t> cores = getCores();
  if (cores.size() % 2 != 0) {
    return emitOpError()
           << "cores must contain an even number of entries (x,y pairs); got "
           << cores.size();
  }
  for (int64_t c : cores) {
    if (c < 0) {
      return emitOpError() << "core coordinates must be non-negative; got "
                           << c;
    }
  }
  return success();
}

LogicalResult ProgramOp::verify() {
  // CoreSetDisjoint invariant: every coordinate claimed by a tt_pk.core_set
  // op in this program's body may appear at most once. Overlap would mean
  // two placements fight for the same Tensix tile.
  llvm::DenseMap<std::pair<int64_t, int64_t>, Operation *> seen;
  WalkResult walk = getBody().walk([&](CoreSetOp cs) -> WalkResult {
    ArrayRef<int64_t> cores = cs.getCores();
    for (size_t i = 0; i + 1 < cores.size(); i += 2) {
      std::pair<int64_t, int64_t> coord{cores[i], cores[i + 1]};
      auto [it, inserted] = seen.try_emplace(coord, cs.getOperation());
      if (!inserted) {
        cs.emitOpError() << "core (" << coord.first << "," << coord.second
                         << ") already assigned to another core_set in this "
                            "program";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return walk.wasInterrupted() ? failure() : success();
}
