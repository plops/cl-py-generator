- this is an experiment to estimate color space conversion
  coefficients using least squares optimization

- i use opencv's color conversion to convert BGR to YCrCb but the
  actual conversion could also be a closed source algorithm

- as long as you can collect enough samples to cover the transform a
  fit of the model should be possible (assuming you can find valid
  good model)
