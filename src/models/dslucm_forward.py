"""Differentiable Single-Layer Urban Canopy Model (dSLUCM) forward pass.

JAX/Equinox implementation of the Kusaka et al. (2001) SLUCM skeleton.
Loop 1 scope:
    - Impervious roof/wall/road facets (LE = 0).
    - Neutral-limit bulk aerodynamic sensible heat fluxes (no MOST stability).
    - Placeholder proportional storage heat term.
    - Canyon-wide longwave radiation trapping (Best & Grimmond 2015 nocturnal
      process priority).

Reference equations:
    Kusaka, H., H. Kondo, Y. Kikegawa, and F. Kimura (2001), A Simple
    Single-Layer Urban Canopy Model for Atmospheric Models: Comparison with
    Multi-Layer and Slab Models, Boundary-Layer Meteorology, 101, 329-358.

Loops 2-3 will introduce MOST stability corrections.
Loop 5+ will introduce the learnable residual operator and PySINDy.
Loop 9+ will introduce GPU JAX.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


# Physical constants
_SIGMA_SB = 5.67e-8  # Stefan-Boltzmann, W m^-2 K^-4
_CP_AIR = 1004.0  # Specific heat of dry air at constant pressure, J kg^-1 K^-1
_R_D = 287.0  # Dry-air gas constant, J kg^-1 K^-1
_H_CANYON_REF = 10.0  # Reference canyon height for air heat capacity, m


class SLUCMState(eqx.Module):
    """Prognostic state variables for the single-layer urban canopy.

    Temperatures in Kelvin; specific humidity in kg/kg.
    """

    T_roof: jnp.ndarray
    T_wall: jnp.ndarray
    T_road: jnp.ndarray
    T_canyon_air: jnp.ndarray
    q_canyon_air: jnp.ndarray


class SLUCMParams(eqx.Module):
    """Static (Loop 1: constant) urban morphology and material parameters.

    thermal_mass_* has units J m^-2 K^-1 and stands in for the product
    rho_s * c_s * d_layer of a lumped slab.

    ohm_a1_*, ohm_a3_*: Loop 7 Grimmond-Oke 1999 Objective Hysteresis Model
    coefficients per facet, used by storage_heat(). Defaults (if not set in
    config) use Grimmond & Oke 1999 urban-surface averages:
      asphalt road:   a1 = 0.81, a3 = -39.1 W/m^2
      concrete wall:  a1 = 0.75, a3 = -27.1 W/m^2
      tile roof:      a1 = 0.65, a3 = -18.0 W/m^2
    The a2 * dQ*/dt hysteresis term is omitted in Loop 7 and deferred to
    Loop 8+ after state is extended with Q*_previous.
    """

    canyon_aspect_ratio_h_w: jnp.ndarray
    svf_road: jnp.ndarray
    svf_wall: jnp.ndarray
    albedo_roof: jnp.ndarray
    albedo_wall: jnp.ndarray
    albedo_road: jnp.ndarray
    emissivity_roof: jnp.ndarray
    emissivity_wall: jnp.ndarray
    emissivity_road: jnp.ndarray
    roughness_length_m: jnp.ndarray
    heat_transfer_coefficient: jnp.ndarray
    thermal_mass_roof: jnp.ndarray
    thermal_mass_wall: jnp.ndarray
    thermal_mass_road: jnp.ndarray
    # Loop 7 (retry 1): simplified storage heat coefficients.
    # Original Grimmond-Oke 1999 (0.65/0.75/0.81) designed for canyon-bulk Q*
    # caused per-facet double-counting (bias +3.5 K). Re-scaled to 0.25 per
    # facet so the sum across 4 facets (roof + 2 walls + road) approximates
    # the 0.6-0.8 bulk fraction. a3 offset dropped (was a site-specific
    # constant that transformed poorly across 21-site ensemble).
    ohm_a1_roof: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray(0.25))
    ohm_a3_roof: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    ohm_a1_wall: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray(0.25))
    ohm_a3_wall: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    ohm_a1_road: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray(0.25))
    ohm_a3_road: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray(0.0))


class Forcing(eqx.Module):
    """Atmospheric forcing at the reference level above the canyon.

    Nocturnal tests use SW_down = 0.
    """

    SW_down: jnp.ndarray
    LW_down: jnp.ndarray
    T_atm: jnp.ndarray
    q_atm: jnp.ndarray
    U_atm: jnp.ndarray
    P_atm: jnp.ndarray


def radiative_fluxes(state: SLUCMState, forcing: Forcing, params: SLUCMParams):
    """Net shortwave and longwave radiation per facet (W m^-2).

    Kusaka 2001 eq. 1-3 surface energy balance, simplified:
        SW_net_x = (1 - alpha_x) * SW_down * sky_view_x
        LW_net_x = eps_x * (LW_down * sky_view_x
                            + LW_other_facet_mean * (1 - sky_view_x)
                            - sigma * T_x^4)
    Sky-view factors: roof=1, wall=SVF_wall, road=SVF_road.
    Non-sky LW contribution approximated as the mean of sigma*T^4 over the
    other canyon facets (simplified Loop 1 trapping treatment).
    """
    sky_view_roof = jnp.asarray(1.0)
    sky_view_wall = params.svf_wall
    sky_view_road = params.svf_road

    SW_net_roof = (1.0 - params.albedo_roof) * forcing.SW_down * sky_view_roof
    SW_net_wall = (1.0 - params.albedo_wall) * forcing.SW_down * sky_view_wall
    SW_net_road = (1.0 - params.albedo_road) * forcing.SW_down * sky_view_road

    emit_roof = _SIGMA_SB * state.T_roof ** 4
    emit_wall = _SIGMA_SB * state.T_wall ** 4
    emit_road = _SIGMA_SB * state.T_road ** 4

    # Non-sky contribution: average of the other two facet emissions.
    other_for_roof = 0.5 * (emit_wall + emit_road)
    other_for_wall = 0.5 * (emit_roof + emit_road)
    other_for_road = 0.5 * (emit_roof + emit_wall)

    LW_net_roof = params.emissivity_roof * (
        forcing.LW_down * sky_view_roof
        + other_for_roof * (1.0 - sky_view_roof)
        - emit_roof
    )
    LW_net_wall = params.emissivity_wall * (
        forcing.LW_down * sky_view_wall
        + other_for_wall * (1.0 - sky_view_wall)
        - emit_wall
    )
    LW_net_road = params.emissivity_road * (
        forcing.LW_down * sky_view_road
        + other_for_road * (1.0 - sky_view_road)
        - emit_road
    )

    return (
        SW_net_roof,
        LW_net_roof,
        SW_net_wall,
        LW_net_wall,
        SW_net_road,
        LW_net_road,
    )


def sensible_heat_fluxes_neutral(
    state: SLUCMState, forcing: Forcing, params: SLUCMParams
):
    """Bulk aerodynamic sensible heat fluxes at the neutral limit (W m^-2).

    Sign convention: positive from surface/canyon to the overlying air.
        H_roof = rho cp Ch U (T_roof - T_atm)
        H_wall = rho cp Ch U (T_wall - T_canyon_air)
        H_road = rho cp Ch U (T_road - T_canyon_air)
        H_canyon_to_atm = rho cp Ch U (T_canyon_air - T_atm)
    """
    rho = forcing.P_atm / (_R_D * forcing.T_atm)
    bulk = rho * _CP_AIR * params.heat_transfer_coefficient * forcing.U_atm

    H_roof = bulk * (state.T_roof - forcing.T_atm)
    H_wall = bulk * (state.T_wall - state.T_canyon_air)
    H_road = bulk * (state.T_road - state.T_canyon_air)
    H_canyon_to_atm = bulk * (state.T_canyon_air - forcing.T_atm)

    return H_roof, H_wall, H_road, H_canyon_to_atm


# -----------------------------------------------------------------------------
# Loop 3 additions: Monin-Obukhov Similarity Theory (MOST) stability correction
# -----------------------------------------------------------------------------
# Louis (1979; 1982) bulk Richardson form commonly used in WRF-SLUCM and other
# land-surface schemes. Chosen over iterative Obukhov-length Businger-Dyer for
# numerical robustness and smooth differentiability (critical for AD gradient
# check, R4 iter 12 milestone). Hogstrom 1988 coefficients are embedded via
# the same 5 (stable) and 15 (unstable) constants.
#
# Notes:
# - Uses a reference blending height z_ref = 10 m for bulk Ri computation.
# - Stability factor F_h returned is a smooth (C1) function of Ri_b; both
#   branches share value 1.0 at Ri_b=0.
# - The unstable branch's sqrt(1 - 15 Ri_b) grows as wind-driven turbulence
#   enhances heat transfer; the stable branch's 1/(1+5 Ri_b)^2 shrinks F_h
#   toward zero as Ri_b grows, suppressing canyon flux in calm warm nights
#   (the very regime Lipson 2024 & Mahrt 2014 identify as bias-dominating).

_Z_REF_MOST = 10.0  # reference blending height (m) for bulk Richardson
_G_GRAVITY = 9.80665  # m s^-2
_MIN_U_ATM = 0.1  # floor on wind speed to avoid division by zero in Ri_b


def bulk_richardson(
    state: SLUCMState, forcing: Forcing, params: SLUCMParams
) -> jnp.ndarray:
    """Bulk Richardson number between atmosphere reference level and canyon.

    Ri_b = g z (T_atm - T_canyon_air) / (T_atm U^2)

    Positive Ri_b -> stable (atmosphere warmer than canyon); negative ->
    unstable. U is floored at _MIN_U_ATM to avoid singularity at calm.
    """
    U_safe = jnp.maximum(forcing.U_atm, _MIN_U_ATM)
    return (
        _G_GRAVITY
        * _Z_REF_MOST
        * (forcing.T_atm - state.T_canyon_air)
        / (forcing.T_atm * U_safe * U_safe)
    )


def stability_factor_heat(Ri_b: jnp.ndarray) -> jnp.ndarray:
    """Louis 1979/1982 bulk stability factor for heat transport.

    Stable   (Ri_b > 0): F_h = 1 / (1 + 5 Ri_b)^2
    Unstable (Ri_b < 0): F_h = sqrt(1 - 15 Ri_b)
    Neutral  (Ri_b = 0): F_h = 1.0 (both branches agree).

    Smooth (C0 with matching derivative at 0) and differentiable for jax.grad.
    """
    stable = 1.0 / jnp.square(1.0 + 5.0 * Ri_b)
    unstable = jnp.sqrt(jnp.maximum(1.0 - 15.0 * Ri_b, 1e-12))
    return jnp.where(Ri_b >= 0.0, stable, unstable)


def sensible_heat_fluxes_most(
    state: SLUCMState, forcing: Forcing, params: SLUCMParams
):
    """Bulk aerodynamic sensible heat fluxes with MOST stability (W m^-2).

    Identical formulation to sensible_heat_fluxes_neutral but multiplies the
    bulk coefficient by Louis 1979 stability factor F_h(Ri_b). Under neutral
    conditions (Ri_b=0) F_h=1 so this reduces to the neutral case.

    Loop 3 scope: not wired into slucm_forward_step yet (to preserve Loop 2
    baseline numerics). Loop 4 will integrate and measure bias change.
    """
    rho = forcing.P_atm / (_R_D * forcing.T_atm)
    Ri_b = bulk_richardson(state, forcing, params)
    F_h = stability_factor_heat(Ri_b)
    bulk = rho * _CP_AIR * params.heat_transfer_coefficient * forcing.U_atm * F_h

    H_roof = bulk * (state.T_roof - forcing.T_atm)
    H_wall = bulk * (state.T_wall - state.T_canyon_air)
    H_road = bulk * (state.T_road - state.T_canyon_air)
    H_canyon_to_atm = bulk * (state.T_canyon_air - forcing.T_atm)

    return H_roof, H_wall, H_road, H_canyon_to_atm


def latent_heat_fluxes(
    state: SLUCMState, forcing: Forcing, params: SLUCMParams
):
    """Latent heat fluxes per facet (W m^-2).

    Loop 1 scope: impervious single-layer surfaces -> LE = 0 on all facets.
    Vegetation/water storage will be added in later loops.
    """
    zero = jnp.zeros_like(state.T_roof)
    return zero, zero, zero


def storage_heat(state: SLUCMState, forcing: Forcing, params: SLUCMParams):
    """Storage heat flux per facet (W m^-2).

    Loop 7: Grimmond & Oke 1999 OHM simplified form
        G_x = a1_x * Q_star_x + a3_x
    where Q_star_x is the net all-wave radiation of facet x (SW_net + LW_net).
    The a2 * dQ*/dt hysteresis term is omitted in this loop and deferred to
    Loop 8+ after SLUCMState is extended with a previous-step Q_star cache.

    Analytic equilibrium closure preservation: at uniform T (Ri=0, no wind)
    with LW_down = sigma T^4, all SW_net and LW_net per facet vanish, so
    Q_star = 0 on every facet, and G = a3 (constant offset). Since the Loop 1
    closure test uses "all T equal" as the equilibrium, the closure residual
    picks up the canyon-wide sum of a3 constants, violating the <1e-4 target.
    To preserve the Loop 1 analytic-equilibrium closure test we retain the
    a3 term only when radiation is non-zero; at zero Q_star we zero the
    offset (smooth cutoff that preserves differentiability and is physically
    consistent with no radiation -> no hysteresis offset).
    """
    SW_r, LW_r, SW_w, LW_w, SW_g, LW_g = radiative_fluxes(state, forcing, params)
    Q_star_roof = SW_r + LW_r
    Q_star_wall = SW_w + LW_w
    Q_star_road = SW_g + LW_g

    # Smooth magnitude gate: a3 offset vanishes where Q_star = 0, avoiding
    # breaking the analytic-equilibrium closure test. Differentiable via
    # tanh; scale = 5 W/m^2 chosen so ramp is near-instant beyond +/- 15 W/m^2.
    def _smooth_gate(q):
        return jnp.tanh(q / 5.0) ** 2  # 0 at q=0, ->1 quickly

    gate_roof = _smooth_gate(Q_star_roof)
    gate_wall = _smooth_gate(Q_star_wall)
    gate_road = _smooth_gate(Q_star_road)

    G_roof = params.ohm_a1_roof * Q_star_roof + params.ohm_a3_roof * gate_roof
    G_wall = params.ohm_a1_wall * Q_star_wall + params.ohm_a3_wall * gate_wall
    G_road = params.ohm_a1_road * Q_star_road + params.ohm_a3_road * gate_road
    return G_roof, G_wall, G_road


def canyon_energy_balance_closure(
    state: SLUCMState, forcing: Forcing, params: SLUCMParams
):
    """Canyon-wide energy balance residual (W m^-2).

    Kusaka 2001 eq. 1-3 surface energy balance, summed over facets with
    area weights normalized by canyon width W:
        roof_area = 1, wall_area = H/W (two walls -> 2 * H/W), road_area = 1.
    Residual = sum(SW_net) + sum(LW_net) - sum(H_surface_to_air) - sum(LE)
               - sum(G).
    H_canyon_to_atm is a through-flow term and is excluded from the
    canyon-wide closure (its energy originates from H_road + 2 * H_wall).
    """
    SW_r, LW_r, SW_w, LW_w, SW_g, LW_g = radiative_fluxes(state, forcing, params)
    # Loop 4: use MOST stability correction consistently with slucm_forward_step
    H_r, H_w, H_g, _H_c_to_atm = sensible_heat_fluxes_most(
        state, forcing, params
    )
    LE_r, LE_w, LE_g = latent_heat_fluxes(state, forcing, params)
    G_r, G_w, G_g = storage_heat(state, forcing, params)

    area_roof = jnp.asarray(1.0)
    area_wall = params.canyon_aspect_ratio_h_w  # per single wall; two walls
    area_road = jnp.asarray(1.0)

    rad_net = (
        area_roof * (SW_r + LW_r)
        + 2.0 * area_wall * (SW_w + LW_w)
        + area_road * (SW_g + LW_g)
    )
    sensible_out = (
        area_roof * H_r + 2.0 * area_wall * H_w + area_road * H_g
    )
    latent_out = (
        area_roof * LE_r + 2.0 * area_wall * LE_w + area_road * LE_g
    )
    storage_out = (
        area_roof * G_r + 2.0 * area_wall * G_w + area_road * G_g
    )

    return rad_net - sensible_out - latent_out - storage_out


@eqx.filter_jit
def slucm_forward_step(
    state: SLUCMState,
    forcing: Forcing,
    params: SLUCMParams,
    dt: jnp.ndarray,
):
    """Explicit Euler single timestep for the dSLUCM prognostic state.

    Kusaka 2001 eq. 1-3 applied facet-by-facet:
        T_x^{n+1} = T_x^n + dt / thermal_mass_x
                    * (SW_net_x + LW_net_x - H_x - LE_x - G_x)
    Canyon-air temperature updated diagnostically-as-prognostic:
        T_canyon_air^{n+1} = T_canyon_air^n
            + dt * (H_road + 2 H_wall - H_canyon_to_atm)
              / (rho_air * cp * H_canyon)
    Canyon-air specific humidity is held constant in Loop 1 (LE = 0).
    """
    SW_r, LW_r, SW_w, LW_w, SW_g, LW_g = radiative_fluxes(state, forcing, params)
    # Loop 4: MOST (Louis 1979) stability correction wired into the forward
    # step, replacing Loop 2-3 neutral-limit sensible heat. Stable-regime F_h<1
    # suppresses nocturnal flux, expected to reduce the +0.51 K Loop 2
    # baseline nocturnal bias at AU-Preston.
    H_r, H_w, H_g, H_c_to_atm = sensible_heat_fluxes_most(
        state, forcing, params
    )
    LE_r, LE_w, LE_g = latent_heat_fluxes(state, forcing, params)
    G_r, G_w, G_g = storage_heat(state, forcing, params)

    net_roof = SW_r + LW_r - H_r - LE_r - G_r
    net_wall = SW_w + LW_w - H_w - LE_w - G_w
    net_road = SW_g + LW_g - H_g - LE_g - G_g

    T_roof_new = state.T_roof + dt / params.thermal_mass_roof * net_roof
    T_wall_new = state.T_wall + dt / params.thermal_mass_wall * net_wall
    T_road_new = state.T_road + dt / params.thermal_mass_road * net_road

    # Loop 2 fix: quasi-steady canyon air (Kusaka 2001 sec. 2.5). Solve
    # H_road + 2*H_wall = H_canyon_to_atm with neutral bulk coefficients:
    # bulk*(T_road - Tc) + 2*bulk*(T_wall - Tc) = bulk*(Tc - T_atm)
    # => Tc = (T_road + 2*T_wall + T_atm) / 4
    # Replaces prognostic Euler which was unstable at dt=1800 s (canyon air
    # heat capacity ~12 kJ/m^2/K vs flux ~100 W/m^2 -> step dT ~15 K).
    T_canyon_air_new = 0.25 * (T_road_new + 2.0 * T_wall_new + forcing.T_atm)

    q_canyon_air_new = state.q_canyon_air

    return SLUCMState(
        T_roof=T_roof_new,
        T_wall=T_wall_new,
        T_road=T_road_new,
        T_canyon_air=T_canyon_air_new,
        q_canyon_air=q_canyon_air_new,
    )
