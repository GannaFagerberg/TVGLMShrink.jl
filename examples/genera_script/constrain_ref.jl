 # --------------------------------------------------
    # Has either constraint changed the proposed step?
    # --------------------------------------------------
    step_limited = b != d[precision_idx]

    if step_limited

        # Required column of prior covariance
        P_col = @view Omega_prior[:, precision_idx]

        # precision_idx-th column of unconstrained
        # posterior covariance
        C_col = P_col - gain * (H_k * P_col)
        Crr   = C_col[precision_idx]

        if !isfinite(Crr) || Crr <= 0
            error(
                "IPLF precision-state variance must be finite and positive."
            )
        end

        # Conditional/projection adjustment
        # of the full state increment
        adjustment = (b - d[precision_idx]) / Crr
        d .+= adjustment .* C_col

        # Enforce precision increment exactly
        d[precision_idx] = b

        mu_updated = mu_iter + d
    end
end