# Question 1

## 1. Architecture diagram

I use one model. Eight series — levels at S1, S2, S3, S4, and T1, rainfall at R1 and R2, and the reservoir release — are aligned on the 10-minute clock, projected into a short list of tokens, mixed by two attention blocks, and read out as a water level at each of S1–S4. The river is not placed on a spatial grid. With a 6-hour history (assumption A1) and a six-step horizon (assumption A2) the token count is L = 8×36 + 6 = 294. Embedding width d = 64 and the depth of two blocks are choices of this design, not facts from the brief.

A station’s own history is not a sufficient input. Two issuances can share the same 36 levels at S2 and differ at S1 or at T1. Those are different rivers, and S2’s forecast has to be allowed to differ. The token list contains the other series, so the two issuances are different inputs. A stack that only ever reads S2 cannot do that, however deep it is.

Reach is set here by the mixer, not by stacking local neighbourhoods. One attention block has a direct connection from every token that remains after the mask to every other such token. A second block does not see farther. It can use a combination the first block has already formed, for example a release and a gauge that have been mixed before the S4 query reads them. I do not count reach in station hops. The stations are unevenly spaced, so the step from S1 to S2 and the step from S3 to S4 are not the same travel time, and neither travel time is constant. A local mixer that saw only the immediate upstream station, and only two lags, would need many stacked blocks before S4 could see the reservoir at a lag of half an hour. Attention buys that path in one block. Depth would be the other way to buy it. I use the direct mixer because the lags themselves change.

Attention is cheap at this size because there are eight series, not a field of sites. The leading cost of one attention block is about 2 L² d multiply-adds: query–key products and the weighted sum of values. With L = 294 and d = 64 that is about 1.1×10⁷ multiply-adds per block, and about twice that for the two encoder blocks. The readout is 24 queries against the same 294 keys, which is smaller. A 64×64 grid over the same 36 times would be 4096×36 tokens, and the quadratic term would be on the order of 10⁵ times larger. The 10-minute cycle is limited by when the gauges arrive, not by this forward pass. The count is from the shapes above. It is not a measured runtime.

A path in the wiring is not the same thing as the model using that path. After the mask is applied, the S4 query has a direct path to every lag of S1, of T1, of both gauges, and of the release. The weights on that path are learned. They can sit on S4’s own recent tokens, in which case an upstream rise is present in the input and does not change the forecast. They can also sit near the uniform weight over the tokens that remain, which is 1/294 when the window is complete, in which case the upstream token is diluted by every other token. The upstream bias in the score is only a preference. Its coefficient is learned and can go to zero, and then the preference is gone as well. This draft does not train the model, so it does not show that the weights follow the river.

```text
Issuance t.  Level clock Δ = 10 min.
H = 36 history bins (A1).   F = 6 leads, +10 … +60 min (A2).

SOURCES                          RAW SHAPE                         PROJECTION
Levels S1 S2 S3 S4 T1            (B, 5, H) value                   W_level
  observed-mask                  (B, 5, H)                         shared
  S3−S2 residual, 0 except S3    (B, 5, H)                         → (B, 5, H, d)
Rain R1 R2, after alignment      (B, 2, H) value + (B, 2, H) mask  W_rain → (B, 2, H, d)
Release history                  (B, 1, H) actual                  W_rel
                                 (B, 1, H) schedule then in force  → (B, 1, H, d)
                                 (B, 1, H) actual-observed mask
Announced future release         (B, 1, F) schedule                W_rel,fut
                                 (B, 1, F) known-at-t mask         → (B, 1, F, d)

Static, no time axis: along-river distances, gauge-to-station
distances, upstream matrix U (8×8), eight series embeddings.

ALIGN (before the projections)
  Five level series share one 10-minute clock (A8).
  Rain keeps its own timestamps and is binned onto that clock (A6).
  Releases are held constant between effective times (A7).
  History bins are t−(H−1)Δ … t.  Horizon bins are t+Δ … t+FΔ.
  Future rain and future levels are not tokens (A13).

RESHAPE AND COMBINE
  Flatten (series, time) → tokens.  Concatenate on the token axis.
  L = 5H + 2H + H + F = 8H + F = 294.
  Add a series embedding and a lag embedding (history) or a lead
  embedding (future release).  Token tensor Z0: (B, L, d), d = 64.

BLOCKS
  Encoder block 1: attention over all L tokens, then a feed-forward layer.
  Encoder block 2: same, separate weights.  Output Z2: (B, L, d).
  Score bias: upstream indicator from U, and a function of distance.
  Readout: one query per (station, lead), 4×6 = 24 queries.
  Each query attends to Z2.  A linear map produces one standardized level.
  Invert the per-station training scale and issue the level.

FORECASTS
  Ŷ : (B, 4, 6)
  station axis (S1, S2, S3, S4), lead axis (+10, +20, +30, +40, +50, +60) min.
  T1, R1, R2, and the release have no forecast head.
```

The feed-forward layer inside each block has hidden width 4d and a GELU nonlinearity. The query for lead h at station s is that station’s projected token at time t, a learned lead vector, and one staleness scalar: how many bins since that station last reported. If the token at t is missing, the token part is the latest observed token of that station. If the whole window is missing, the token part is the series embedding. Keys and values are Z2. A history token whose mask is 0, including a carried rain bin, is kept so that L stays fixed, and its attention score is set to −∞ so it receives no weight. The last real token of that series remains a normal key.

## 2. Assumptions

The following are given, and I do not treat them as assumptions. The reservoir releases into the main river, upstream of S1. Monitored tributary T1 joins between S1 and S2. An unmonitored tributary joins between S2 and S3. Stations are unevenly spaced, and travel times can change with river conditions. Levels at S1–S4 and T1 are recorded every ten minutes. Rainfall at R1 and R2 comes with timestamps and gauge locations. Recorded releases, and release schedules already announced at issuance, come with announcement times and effective times. Station locations, connectivity, and distances along the river are available. Precise travel times, unmonitored inflows, future rainfall, and future levels are not available at issuance.

The PDF does not state a history length. It does not state whether “one hour ahead” is a single level at +60 minutes or the levels on the steps inside the hour. Record length, reporting delay, and measurement quality are unspecified. Rainfall is not stated to be on the 10-minute level clock. The figure is not to scale and is not the distance table.

**A1. History length.** The input window is H = 36 bins, six hours, ending at the issuance time. Six hours is several times the one-hour horizon, which leaves room for a wave slower than the forecast itself, and 36 steps keep L near 300. That is a choice of window, not a claim that six hours covers every wave. This sets which upstream observations exist as tokens. A travel time longer than six hours is invisible except through the level it has already produced inside the window. I would change H first if measured travel times (section 6) showed that six hours cuts off ordinary waves.

**A2. Shape of the one-hour output.** The issued forecast is six levels at each of S1–S4, at +10, +20, +30, +40, +50, and +60 minutes. The service sentence says “one hour ahead” and does not choose between a single +60 minute level and the path inside the hour. I issue the path because the archive and the issuance cycle are both 10 minutes, and a flood desk uses whether the level is rising through the hour. The +60 minute slice is the level the service names. This assumption sets Ŷ to shape (B, 4, 6) and puts six terms in the loss. A single +60 minute head would be a different model.

**A3. Record length.** I assume the training archive covers at least one year of 10-minute operation, including wet and dry periods and more than one release schedule. Inference does not read the year. It reads the window in A1. A year is long enough for seasonal level and for a range of lags to appear at least sometimes. A shorter archive would leave high-flow lags thin. If the real archive is shorter, I would train on what exists and widen the limitation in section 6, not change the tensors.

**A4. Ordinary reporting delay.** On a normal issuance, the bin that closes at t has arrived for all five level series, both rain gauges, and the actual release. The PDF does not state a delay. This assumption lets the history axis end at t. When it fails, those bins stay masked and the issuance still goes out (section 5). I do not hold the forecast for a late packet.

**A5. Measurement quality.** Errors on a series are unbiased and small beside a rise that matters for flood preparedness. No standard deviation is given, and I do not invent one. There is no separate denoiser. Noise in the recorded targets remains in the loss, so the fit cannot be tighter than the gauges. A stuck or drifted gauge would be learned as if it were the river.

**A6. Rainfall cadence.** Rain is not on the level clock unless a timestamp happens to fall on a bin boundary. Each reading is placed in the 10-minute bin that contains its timestamp. A bin with no reading repeats the last real reading and is marked carried, not freshly observed. This gives rain the same H axis as the levels, which is what lets one token list hold both. A carried value only fills the bin so the tensor stays dense. It is not a claim that rain continued at that rate, and section 4 gives that bin no attention weight. The last real reading remains the rain token that can be used. Sub-bin timing inside the 10 minutes is discarded.

**A7. Release time base.** Recorded releases and announced effective times are piecewise constant between events. I place each event in the bin that contains its effective time and hold the value until the next event. If more than one announcement for the same effective time is already public at issuance, I use the latest of them. Operations are scheduled as holds and changes, which matches a piecewise-constant series better than a native 10-minute sensor. The cost is that a change inside a bin is treated as in force for the whole bin.

**A8. Shared level clock.** S1, S2, S3, S4, and T1 use the same 10-minute boundaries. The brief says each is recorded every ten minutes and does not say the clocks match. A shared clock is what an authority network usually aims at, and it is what makes one lag index mean the same time at every station. An offset larger than a bin would put the wrong upstream level on that lag.

**A9. Downstream transport, no fixed speed.** Water from the reservoir, from T1, and from the unmonitored tributary moves downstream along the given connections. It does not move upstream. Travel time is not distance divided by a constant speed. Distances are static features in the attention score. Past levels at downstream stations remain legal inputs: they were measured before issuance and they describe the profile that is already in the river. This assumption writes the upstream matrix U in section 4 and stops me from inserting a delay equation. It does not delete downstream history.

**A10. The target is level.** No rating curve is supplied, so I map levels, rain, and releases directly to future levels. Storage, slope, and roughness are not states. The service asks for level. A release in volume per time and a level in length stay in separate projections; any conversion between them is learned. Within each series I take the stored unit as constant over the archive. I standardize each series with its training mean and standard deviation and invert that scale before the issued number leaves the model. A later change of units would require refitting those two constants.

**A11. No assigned rain catchment.** R1 is on the upper hillside and R2 is farther downstream. The figure is not to scale and does not say which gauge drains to which reach. I do not assign one. Both gauges are tokens at every issuance, and gauge-to-station distance is a feature (A16). The learned scores decide the association. A wrong hard-wired catchment would be worse than a weak learned one, because I was not given the map.

**A12. The unmonitored tributary is latent.** The junction between S2 and S3 is real and has no series. I do not invent a gauge. I do feed the observed difference between S3 and S2 on the history, masked when either level is missing, as an extra channel that is zero on the other level series. That difference mixes the tributary, the channel, and gauge error. It is computable from given levels. S3 and S4 can respond to inflow that has already reached S3. Inflow that arrives only after issuance is still unseen. S1 and S2 are upstream of that junction, so this channel is not a driver for their targets.

**A13. What may sit on the horizon.** Future rainfall and future levels are not inputs. A release value is written on a horizon bin only when its announcement time is at or before issuance and its effective time falls in that bin. This is the information constraint drawn in section 1. Filling the horizon with a rainfall forecast, or with the release that later actually occurred, would train on something the desk did not have.

**A14. Training respects the same clock.** For an issuance at t drawn from the archive, inputs are restricted by A13, and the target is the level later recorded at t+Δ, …, t+6Δ. Each such t is one row of the batch. This stops the fit from using rain or announcements that arrived after t. It is also why the future-release tensor depends on the announcement time and not only on the effective time.

**A15. Point forecast, squared error.** The training objective is the mean squared error of the standardized level, averaged over the four stations, the six leads, and the issuances in the batch. Missing targets are left out of the average rather than filled. No loss is specified in the brief. Squared error is a plain point-forecast loss and it charges more for a large miss. The issued number is a conditional mean under this loss, not a flood probability and not an upper envelope. A quantile head or a simulator is optional in the brief and is not part of this design.

**A16. Two meanings of distance.** Along-river distances supplied by the authority are used among the reservoir, S1–S4, and T1. Gauge-to-station distance is the straight-line distance between the gauge coordinates and the station coordinates, because the brief gives gauge locations and river distances, not a path from the hillside to the channel. Both enter only through φ in the score. Neither is turned into a travel time.

## 3. Task and data table

An issuance is computed when a 10-minute bin closes. Call that time t. The history window is the H = 36 bins from t−35Δ to t, with Δ = 10 minutes (A1). The horizon is one hour: leads h = 1, …, 6 at times t+hΔ (A2). The outputs are the six future levels at S1, S2, S3, and S4. T1 is an input and not a target. The batch axis B is the number of issuance times in a training batch. In operation, B = 1.

The table uses the shapes after alignment and before the width-d projection. Standardization (A10) is applied to value columns and inverted on Ŷ. Masks stay in {0, 1}.

| Name | Shape | Axes | Time coverage | At issuance |
| --- | --- | --- | --- | --- |
| Level history | (B, 5, H) | batch; series S1, S2, S3, S4, T1; lag | t−(H−1)Δ … t | Known. Past only. |
| Level mask | (B, 5, H) | same | same bins; 1 if that bin was observed | Known. |
| S3−S2 residual | (B, 5, H) | same series axis; only the S3 row is nonzero | same bins; 0 where either level is missing | Known. Constructed from the two levels (A12). |
| Rain, aligned | (B, 2, H) | batch; gauge R1, R2; lag | bins on the level clock; source timestamps may lie inside the bin | Past readings only, after A6. |
| Rain mask | (B, 2, H) | same | 1 if the bin contains a real reading; 0 if the value was carried forward | Known. |
| Actual release | (B, 1, H) | batch; one release series; lag | bins ending at t, held by effective time (A7) | Known recorded actuals. |
| Schedule on the history | (B, 1, H) | same | the announcement that was already public for that past bin | Known from the announcement log. |
| Actual-release mask | (B, 1, H) | same | 1 if an actual value was recorded in the bin | Known. |
| Announced future release | (B, 1, F) | batch; one release series; lead | t+Δ … t+6Δ | Known only where announcement time ≤ t (A13). |
| Future-release mask | (B, 1, F) | same | 1 if a public announcement covers that lead | Known. |
| Upstream matrix U | (8, 8) | source series; destination series | none; static | Known from connectivity (A9). |
| Distances | (8, 8) | series; series | none; static | Known. River distances, and straight-line gauge distances (A16). |
| Forecast Ŷ | (B, 4, 6) | batch; station S1–S4; lead h = 1…6 | t+Δ … t+6Δ | Not known at issuance. Training target only. Never an input. |

Series order inside every history block is S1, S2, S3, S4, T1, R1, R2, release. That order is what the flatten in section 1 uses, so a reshape does not swap a gauge with a station. Staleness τ_s is not a row in this table. It is computed from the level mask when the query is built.

Alignment across places is a shared time index, not a resampled map. Stations stay eight discrete series. I do not bin the river into equal kilometres. Uneven spacing stays in the distance matrix. Alignment across sampling times is the binning in A6, A7, and A8: levels already share a clock, rain is moved onto that clock from its own timestamps, and releases are stepped onto that clock by effective time. Tokens from different series that share a lag index refer to the same 10-minute bin. Future release tokens use the lead index, not the lag index, so a horizon bin is not stored as if it were a past observation.

At issuance the model may see every row of the table except Ŷ, and the future-release row only through announcements already public. It may not see future rain, future levels, the unmonitored inflow, a measured travel time, or a release announcement whose announcement time is still in the future.

## 4. Notation, core equations, training objective, justification

Δ = 10 minutes. Issuance time t. Lag ℓ = 0, …, H−1 means time t − ℓΔ, so ℓ = 0 is the issuance bin. Lead h = 1, …, 6 means time t + hΔ. Series index i runs over the eight series in the order in the table. y_s is the level at station s ∈ {S1, S2, S3, S4}. Batch index is omitted; every learned tensor below has a leading batch axis.

U is 8×8 with U_{j,i} = 1 when series j is upstream of series i, and 0 otherwise (A9):

- Into S1: the release only.
- Into S2: the release, S1, and T1.
- Into S3: the release, S1, T1, and S2.
- Into S4: the release, S1, T1, S2, and S3.
- Into T1, R1, R2, and the release: no series in this token list is upstream. Rain is not given an upstream bit (A11).

A station is not upstream of itself. Own-history tokens therefore get no upstream bonus. They still match through the query, which contains that station’s latest token.

φ(d) = −log(1 + d / d0), where d0 is the median of the static distances, computed once. Nearer pairs score higher before λ_dist is applied. λ_up and λ_dist are scalars, learned separately in the encoder and in the readout.

**Projection.** For a level token, the raw vector is the standardized level, the mask, and the residual channel of A12. Rain uses standardized rain and its mask. A history release token uses standardized actual, standardized schedule, and the actual mask. A future release token uses the standardized announced value and the known mask. Staleness is not another stored series. It is one scalar on the station query, defined in the readout. Each group has its own matrix:

e = W_{group} x + a_i + b_time,

with W_level, W_rain, W_rel, and W_rel,fut mapping into R^d, a_i a series embedding, and b_time a lag embedding or a lead embedding. Masked values are set to 0 after standardization, which is the training mean in the original unit. The token list is the concatenation in section 1, Z^{(0)} ∈ R^{L×d}.

**Encoder block.** For n = 1, 2, with separate weights, and LN layer normalisation:

Q, K, V = LN(Z^{(n−1)}) W^Q, W^K, W^V,

score(p, q) = (Q_p · K_q) / √d + λ_up U_{series(q), series(p)} + λ_dist φ(d_{q,p}).

A key whose history mask is 0, including a carried rain bin, has score −∞ and drops out of the softmax. The token is still stored, so L does not change. Attention is the softmax over the remaining keys. The block output is the residual attention plus a residual feed-forward layer of width 4d. There is no time mask inside the observed history: every remaining history bin is already past at t. Future release tokens with known mask 1 are visible because those releases are already announced (A13). A future bin with known mask 0 is also −∞. There are no future level tokens and no future rain tokens.

**Readout.** Let z_{s,0} be the projected token of station s at ℓ = 0, or the latest observed token of s, or the series embedding if none exists, as in section 1. Let τ_s be the number of bins since s last had a mask of 1, or H if it never did inside the window. The query for lead h is

q_{s,h} = W_q [ z_{s,0} ; e_h ; τ_s ].

Keys and values come from Z^{(2)}. The score adds λ_up^{read} U_{j,s} + λ_dist^{read} φ(d_{j,s}) for a key of series j. The weighted values pass through a linear map to a standardized level, and A10 inverts the scale to ŷ_{s,h}(t).

**Training objective.** Issuances are drawn from the archive under A14. m_{s,h}(t) = 1 when the target level was recorded:

L = mean over the batch of [ Σ_{s,h} m_{s,h} (ŷ^{std}_{s,h} − y^{std}_{s,h})^2  /  Σ_{s,h} m_{s,h} ].

The denominator is the count of observed targets at that issuance, so a missing label does not become a zero level. Each issuance contributes once.

**Why this layout.** Separate projections exist because the units differ (A10). One token list exists so that S2’s query can see S1 and T1, which a single-series temporal model cannot. Two blocks, with attention rather than a local spatial mix, are the reach argument in section 1. Lead-specific queries exist so that the +10 minute forecast and the +60 minute forecast can put weight on different lags. That is the mechanism for a travel time that is not one fixed offset. The direct head, rather than a rollout that feeds predicted levels back in, matches a six-step horizon: the announced release stays an input, and a predicted level is not written back as if it had been measured.

**How the drivers reach each forecast.** These are routes the wiring allows. Section 1’s warning still applies: a route is not evidence that the trained weights use it.

- S1. Own levels carry the local rise. The release is the upstream boundary: past actuals are water already let out, and the announced values on leads 1…h are water the operator has committed to. S1 can move inside the hour only when the outlet-to-S1 travel time fits in that hour. That time is not given, so the lag is learned. T1 joins downstream of S1, so U does not prefer T1 for this query. Rain reaches S1 only through whatever association the scores learn (A11), and only from rain already fallen (A13).
- S2. The extra drivers, relative to S1, are S1’s levels and T1’s levels, at lags that can cover the two paths into the junction between them. A model that saw only S2 would return the same forecast for two windows that differ only upstream. The release reaches S2 through the same learned lag, typically later than it reaches S1. Rain is the same learned association, from past rain only.
- S3. Drivers are S2, and through S2 the stem above it, plus rain and the release. The unmonitored tributary has no token. Its trace, once it has reached S3, is in S3’s own levels and in the S3−S2 residual (A12). Water that enters that tributary after t is not in the input.
- S4. The same upstream set is visible in one block, without a hop limit. S3 is the nearest upstream level. The unmonitored tributary reaches S4 only after it has passed S3, so S4’s warning is S3’s recent rise, delayed again by the S3–S4 travel time. That delay is not a constant (A9). Lead-specific queries are where a longer lead can look further back.

**What is fixed and what is learned.** Fixed from the river and from the clock: the series list, U, the absence of future rain and future levels, the rule that an announcement enters only if it is already public, piecewise-constant binning, and the decision to forecast level rather than discharge. Those are A6–A16 and the diagram. Learned: all projection matrices, the embeddings, both encoder blocks, the readout, and the four bias scalars. Learned, and not guaranteed by U: which lag receives the weight, whether T1 matters at S2, whether R1 matters at S4, and whether an announced increase moves any station inside the hour. Distance does not set those lags. I do not run a hydraulic simulator beside this model. The physical content is the graph, the information constraint, and the binning. The map from tokens to levels is the fit.

## 5. Four separate treatments

**(i) Missing observations.** A missing level, rain reading, or actual release sets that bin’s value to 0 after standardization and sets its mask to 0. The token stays in the list, so L and the series order do not change when a gauge drops, and the score rule in section 4 gives it no attention weight. The S3−S2 residual is masked if either station is missing, so a gap is not turned into a false inflow. Targets that are missing are omitted from the loss in section 4. They are not filled with a neighbouring station. At runtime a missing S1 does not block the S4 forecast: the S4 query still attends to whatever of S2, S3, T1, rain, and the release is present. If a station’s own latest token is missing, its query falls back as in section 1. A long gap is a run of masked tokens. I do not interpolate across it with a travel time I was not given.

**(ii) Delayed observations.** A bin that has not arrived by t is a missing bin under (i). The extra quantity is τ_s in the readout: how many bins since that station last reported. Rain and release delays have no forecast query; their last real token remains a key, and later carried or empty bins are scored −∞. The forecast is issued on the 10-minute mark with the masks as they stand. I do not wait, under A4’s failure mode, for a packet that missed the bin. The next issuance is ten minutes later. If the late packet has arrived, it fills the earlier bin in the new window and the masks change. A delay longer than H leaves that series with no observed token in the window; a station query then uses the series embedding and τ_s = H, and the other series still support their own heads.

**(iii) Announced future releases.** For each lead h I look in the announcement log for announcements with announcement time ≤ t and effective time in the bin t+hΔ. The latest such announcement is the value in the future-release tensor, and the known mask is 1 (A7, A13). If none exists, the value is 0 after standardization and the known mask is 0. A flat schedule is still an input: the mask is 1 and the value is the held release, so “no change” is different from “no announcement.” Announcements whose announcement time is after t are ignored for this issuance, including ones that revise the next hour. Releases announced for later than t+60 minutes are outside this horizon and are not written into these six bins. They can enter a later issuance. The four heads read these tokens through the same attention as the history. The release is not a target.

**(iv) Scheduled versus actual releases.** On history bins the actual-release channel is what was recorded, and the schedule channel is what had been announced for that bin. Both are inputs, so the recent gap between them is visible at t. The future channel is only the schedule. The actual release after t is not known and is not written in, during training or in operation (A13). The loss compares the forecast with the level that actually occurred, which depended on the actual release. When operators depart from the schedule, that departure is an error the issuance at t cannot remove. The following issuance, ten minutes later, observes the actual in the history and can change the forecast. I do not overwrite the announced future with a guessed actual, and I do not train with the actual future release in its place. That would credit the model for seeing a decision that had not been taken.

**A forecast every ten minutes.** Each issuance is one forward pass of the network in section 1: two attention blocks over 294 tokens of width 64, then 24 readout queries. The attention term is about 2×10⁷ multiply-adds for the two blocks, and the two feed-forward layers are of the same order. Weights are fixed between issuances. The window slides by one bin. There is no simulator step and no iterative solve that has to converge inside the cycle. The shapes in section 3 stay the same when masks change, so a missing gauge does not rebuild the computation. Ten minutes is far longer than a pass of this size on a desktop machine. I report the count from the tensor shapes. I do not report a benchmark.

## 6. Uncertainties and limitations

**Unmonitored tributary.** Impact: S3 and S4 can rise or fall for a reason that is not in S2, not in T1, not in the release, and not in R1 or R2. S1 and S2 are upstream of the junction, so that inflow is not a driver for their targets. If the tributary responds to rain that neither gauge sees, the one-hour error sits at S3 and S4. Until the extra water reaches S3, the input has no trace of it. After it reaches S3, later issuances can carry that rise toward S4, late by the S3–S4 travel time. The squared-error fit averages over the tributary events in the archive, so a quiet hour and a flash hour that look the same at S2 receive similar forecasts. Response: the residual channel (A12), S3’s own history, and a direct path from those tokens to the S4 query. What remains unresolved is inflow that has not reached S3 by t, and the part of S3−S2 that is channel storage or gauge error rather than the tributary. The path to S3’s recent tokens can exist and still be ignored, which is the same weight problem as in section 1.

**Changing travel times.** Impact: a peak can be early or late. The error accumulates toward S4. Squared error charges a peak that has the right height and the wrong 10-minute bin. A single fixed lag, shared by every issuance, is the wrong response on a river whose travel time changes. Response: attention over every lag in the six-hour window, lead-specific queries, and distance as a feature rather than a speed (A1, A2, A9). The window is what limits the longest lag. What remains unresolved is a travel time longer than H, and a change of speed during the future hour that the current levels, rain, and schedule do not indicate. Here again the direct path to every lag can exist while the weights collapse onto one lag, or onto each station’s own latest token. I have not trained the model, so I have no attention map to show. A map that did not move when the river rose would mean the variable-lag path was unused.

**Other limits of this design.** The issued number is a point level under A15. The tributary and the schedule-versus-actual gap are uncertainties that a single number hides. I do not attach an interval. There is no hydraulic residual check, because there is no rating curve and no simulator. H = 36 and the six-step head are assumptions A1 and A2; a different reading of the service sentence would change Ŷ. Rain that starts after t is invisible by A13. Sub-bin timing is discarded by A6 and A7. Standardization constants are part of the deployed model (A10).

**Additional information.** The system runs with the inputs in section 3 if none of the following is provided.

1. A level or discharge gauge on the unmonitored tributary, even at a coarser step than 10 minutes. It would be a ninth series: its own projection, its own row of U into S3 and S4, and H extra tokens. S3 and S4 could then see the inflow before it reaches S3. Without it, both heads still issue a level from the monitored series, with the residual error described above.
2. A small set of measured travel times on each main reach, at a low stage and at a high stage (dam to S1, S1 to S2, T1 to the junction, S2 to S3, S3 to S4). They would check whether H = 36 cuts off ordinary waves, and they would be an external check on which lags receive weight. They are not required as an input. The forecast does not look up a travel-time table.
3. A catchment statement or polygon for R1 and for R2. It would replace A11’s open association with a known one, or add a prior on U for the two gauges. Without it, both gauges remain tokens and the scores stay learned.
4. A stated precision or error bar for each level series, each gauge, and the release. It would turn A5 into a variance and could reweight the loss toward the cleaner instruments. Without it, each station contributes through its own standardized squared error, and a noisy gauge is treated as a clean one.

A rainfall nowcast is not one of these requests. The service is specified to run without future rain. If a nowcast later existed, it would be a new token group on the six horizon bins, with its own mask, under the same readout. I do not leave an empty rain forecast in the model and call it dry weather.
