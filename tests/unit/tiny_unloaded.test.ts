/**
 * Before the trained weights are provided (the browser, until its fetch
 * lands) the library says so instead of running on nothing.
 */
import { describe, expect, it } from "vitest";

import { hasTrainedWeights, trainedModelWeights } from "@/lib/num/tiny";

describe("tiny model before the trained weights arrive", () => {
  it("reports them missing and refuses to run", () => {
    expect(hasTrainedWeights()).toBe(false);
    expect(() => trainedModelWeights()).toThrow(/not loaded/);
  });
});
