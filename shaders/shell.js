// Brain / VNC glass: a fresnel rim only (faces seen edge-on glow). Kept far below the bloom threshold so it never blooms.
export const shellVertex = /* glsl */ `
varying vec3 vN;
varying vec3 vV;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}
`;

export const shellFragment = /* glsl */ `
uniform vec3 uColor;
uniform float uAlpha;
uniform float uFade;
varying vec3 vN;
varying vec3 vV;
void main() {
  float ndv = abs(dot(normalize(vN), normalize(vV)));
  float rim = pow(1.0 - ndv, 2.5);
  gl_FragColor = vec4(uColor * rim * uAlpha * uFade, 1.0);
}
`;
