// Turns per-frame class probabilities into one discrete decision per threat.
// Fires when the same non-"none" class has prob > p for `frames` consecutive frames; then stays silent until reset().

export function createDecider({ classes, p, frames }) {
  const noneIndex = classes.indexOf("none");
  let streakClass = -1;
  let streak = 0;
  let fired = false;

  return {
    update(probs) {
      if (fired) return null;
      // the most probable class above p (unique whenever p >= 0.5)
      let best = -1;
      for (let c = 0; c < classes.length; c++) if (probs[c] > p && (best < 0 || probs[c] > probs[best])) best = c;
      if (best < 0 || best === noneIndex) {
        streakClass = -1;
        streak = 0;
        return null;
      }
      streak = best === streakClass ? streak + 1 : 1;
      streakClass = best;
      if (streak < frames) return null;
      fired = true;
      return { classIndex: best, name: classes[best] };
    },
    reset() {
      streakClass = -1;
      streak = 0;
      fired = false;
    },
  };
}
