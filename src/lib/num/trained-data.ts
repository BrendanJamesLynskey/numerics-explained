/**
 * The trained weights for server code and tests: imports the vendored file
 * and hands it to the tiny model (`provideTrainedWeights`). Never import
 * this from a client component; the widget fetches the same file instead.
 */
import file from "../../../public/data/trained-weights.json";

import { provideTrainedWeights, type TrainedWeightsFile } from "./tiny";

provideTrainedWeights(file as unknown as TrainedWeightsFile);

/** Optimiser steps behind the weights (Training's default run). */
export const TRAINED_STEPS: number = file.steps;
