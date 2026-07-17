# Account Creation Runbook — AI/ML Research Shorts

Step-by-step, **pre-auth** procedure to create the three social accounts fully
detached from personal identity. Follow in order. Nothing here publishes or
authenticates the pipeline — it only stands up the accounts.

> Prereqs: a dedicated brand email (e.g. `researchfastlane@<provider>`) and one
> virtual/SMS-receiving number (rental service) that can receive TikTok/IG
> verification codes. Do NOT use any personal email/phone/device.

## 0. Isolation environment (do this first)
- Use a **dedicated browser profile** (or anti-detect browser) separate from
  your personal one. Clear cookies; fresh fingerprint.
- Prefer a separate machine/VM or at minimum a distinct browser user profile +
  VPN/clean IP. TikTok links accounts by device ID + IP + fingerprint.
- Never sign into these accounts from your personal phone/app.

## 1. Brand email + number
- [ ] Create the brand email; enable 2FA (authenticator app).
- [ ] Acquire the virtual number; confirm it can receive SMS for TikTok + IG.
- [ ] Store both in the shared secrets manager (NOT the repo).

## 2. YouTube — Brand Account (custom name)
1. Create a NEW Google account using the brand email (non-Gmail login path).
2. Go to YouTube → profile → **Create a channel** → **Use a custom name**.
3. Name: `Research in the Fast Lane`; handle `ResearchInTheFastLane`.
4. Upload avatar (`docs/brand/avatar.png`) + banner (`docs/brand/banner_youtube.png`).
5. Set bio (see `brand_spec.md`). Enable 2FA on the Google account.
6. Note: this Brand Account is separate from any personal Google identity.

## 3. TikTok — standard account
1. In the isolated browser, go to TikTok sign-up → **Use phone or email**.
2. Use the brand email; verify with the virtual number SMS code.
3. Username `@researchfastlane`; set bio + avatar (`docs/brand/avatar_tiktok.png`).
4. Immediately add the brand email as recovery + enable 2FA (authenticator).
5. Do NOT switch to/from any other TikTok account on this environment.

## 4. Instagram — business profile
1. Sign up with the brand email (separate from personal FB/IG).
2. Switch to a **Professional / Business** account for insights.
3. Username `@researchfastlane`; bio + avatar (`docs/brand/avatar_instagram.png`)
   from `brand_spec.md`.
4. Verify phone (virtual number) + enable 2FA.

## 5. Post-creation checklist
- [ ] All three use the brand email + virtual number (no personal contact).
- [ ] 2FA enabled on every account (authenticator app).
- [ ] Recovery contacts stored in secrets manager.
- [ ] Avatar + banner + bios set and consistent with `brand_spec.md`.
- [ ] Accounts NOT logged into from personal devices.
- [ ] No cross-interaction between the three accounts.

## 6. Hand-off to pipeline auth
Only after steps 1–5: run the pipeline's OAuth bootstrap (separate step, not
part of this runbook):
```
python3 -m auto_ingest.shorts.cli publish auth youtube
python3 -m auto_ingest.shorts.cli publish auth tiktok
python3 -m auto_ingest.shorts.cli publish auth instagram
```
Each writes tokens to `~/.config/auto-ingest/` — keep that directory backed by
the same secrets manager.

## Guardrails
- One email + one number per account (TikTok blocks reuse).
- Isolation > convenience: a linked/banished account risks the others.
- Personal accounts stay completely out of this flow.
