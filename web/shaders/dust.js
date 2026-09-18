// Whole-CNS context: every neuron centroid as a 1 px point at a few percent alpha.
export const dustVertex = /* glsl */ `
uniform float uPointSize;
void main() {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = uPointSize;
}
`;

export const dustFragment = /* glsl */ `
uniform vec3 uColor;
uniform float uAlpha;
uniform float uFade;
void main() {
  gl_FragColor = vec4(uColor * uAlpha * uFade, 1.0);
}
`;
