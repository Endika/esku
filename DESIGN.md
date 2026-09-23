---
name: Esku
description: Sign-by-sign LSE recognition in a system-camera viewfinder; the camera is the app.
colors:
  shutter-violet: "#7c3aed"
  shutter-violet-hover: "#8b5cf6"
  signal-yellow: "#ffd60a"
  viewfinder-bg: "#000000"
  viewfinder-ink: "#ffffff"
  viewfinder-muted: "rgba(255, 255, 255, 0.72)"
  viewfinder-soft: "rgba(255, 255, 255, 0.52)"
  viewfinder-faint: "rgba(255, 255, 255, 0.34)"
  viewfinder-line: "rgba(255, 255, 255, 0.16)"
  viewfinder-chip: "#1c1c21"
  viewfinder-well: "#000000"
  viewfinder-focus: "#ffd60a"
  viewfinder-hover: "rgba(255, 255, 255, 0.08)"
  viewfinder-bg-light: "#ffffff"
  viewfinder-ink-light: "#131318"
  viewfinder-muted-light: "rgba(19, 19, 24, 0.7)"
  viewfinder-soft-light: "rgba(19, 19, 24, 0.6)"
  viewfinder-faint-light: "rgba(19, 19, 24, 0.28)"
  viewfinder-line-light: "rgba(19, 19, 24, 0.14)"
  viewfinder-chip-light: "#f1f1f4"
  viewfinder-well-light: "#eef0f3"
  viewfinder-focus-light: "#131318"
  viewfinder-hover-light: "rgba(19, 19, 24, 0.05)"
  video-black: "#000000"
  on-video-ink: "#ffffff"
  on-video-soft: "rgba(255, 255, 255, 0.52)"
  on-video-faint: "rgba(255, 255, 255, 0.34)"
  on-video-scrim: "rgba(0, 0, 0, 0.58)"
  sheet-bg: "#0b0a12"
  sheet-surface: "#16141f"
  sheet-raised: "#211e2e"
  sheet-border: "rgba(255, 255, 255, 0.1)"
  sheet-text: "#f5f3ff"
  sheet-text-muted: "#aaa5c2"
  sheet-text-soft: "rgba(245, 243, 255, 0.12)"
  sheet-danger: "#f87171"
  sheet-bg-light: "#f6f5fa"
  sheet-surface-light: "#ffffff"
  sheet-raised-light: "#f0eef7"
  sheet-border-light: "rgba(24, 20, 45, 0.12)"
  sheet-text-light: "#17142b"
  sheet-text-muted-light: "#57526f"
  sheet-text-soft-light: "rgba(23, 20, 43, 0.1)"
  sheet-danger-light: "#c0262d"
  part-hands: "#a78bfa"
  part-face: "#22d3ee"
  part-neck: "#f472b6"
  part-torso: "#fbbf24"
  part-arms: "#4ade80"
typography:
  display:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "2rem"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.03em"
  live-guess:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "1.85rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.04em"
  transcript:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "1.4rem"
    fontWeight: 560
    lineHeight: 1.35
  headline:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 700
    letterSpacing: "-0.01em"
  title:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "1rem"
    fontWeight: 650
  body:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "0.92rem"
    fontWeight: 400
    lineHeight: 1.5
  status:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 560
    lineHeight: 1.35
  label:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "0.72rem"
    fontWeight: 600
    lineHeight: 1.3
  figure:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "1.4rem"
    fontWeight: 650
    letterSpacing: "-0.02em"
    fontFeature: "tnum"
rounded:
  bracket: "8px"
  row: "10px"
  chip-compact: "12px"
  radius: "14px"
  pill-status: "18px"
  radius-lg: "22px"
  full: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  gap: "16px"
  lg: "20px"
  xl: "24px"
components:
  shutter-disc:
    backgroundColor: "{colors.shutter-violet}"
    textColor: "{colors.viewfinder-ink}"
    rounded: "{rounded.full}"
    size: "72px"
  shutter-disc-hover:
    backgroundColor: "{colors.shutter-violet-hover}"
  shutter-disc-light:
    backgroundColor: "{colors.shutter-violet}"
    textColor: "{colors.viewfinder-ink-light}"
  status-pill:
    backgroundColor: "{colors.viewfinder-chip}"
    textColor: "{colors.viewfinder-ink}"
    typography: "{typography.status}"
    rounded: "{rounded.pill-status}"
    padding: "8px 14px"
  status-pill-light:
    backgroundColor: "{colors.viewfinder-chip-light}"
    textColor: "{colors.viewfinder-ink-light}"
  part-chip:
    backgroundColor: "{colors.viewfinder-chip}"
    textColor: "{colors.viewfinder-muted}"
    typography: "{typography.label}"
    rounded: "{rounded.full}"
    padding: "4px 8px 4px 6px"
  part-chip-on:
    backgroundColor: "{colors.viewfinder-chip}"
    textColor: "{colors.viewfinder-ink}"
  part-chip-light:
    backgroundColor: "{colors.viewfinder-chip-light}"
    textColor: "{colors.viewfinder-muted-light}"
  part-chip-compact:
    rounded: "{rounded.chip-compact}"
    padding: "5px 2px 4px"
  video-frame-idle:
    backgroundColor: "{colors.viewfinder-well}"
    textColor: "{colors.viewfinder-faint}"
  video-frame-idle-light:
    backgroundColor: "{colors.viewfinder-well-light}"
    textColor: "{colors.viewfinder-faint-light}"
    rounded: "{rounded.radius-lg}"
  video-frame-running:
    backgroundColor: "{colors.video-black}"
    textColor: "{colors.on-video-faint}"
  live-guess:
    backgroundColor: "{colors.on-video-scrim}"
    textColor: "{colors.on-video-ink}"
    typography: "{typography.live-guess}"
    rounded: "{rounded.full}"
    padding: "6px 22px"
  chip-button:
    backgroundColor: "transparent"
    textColor: "{colors.viewfinder-ink}"
    rounded: "{rounded.full}"
    padding: "0 16px"
    height: "44px"
  chip-button-hover:
    backgroundColor: "{colors.viewfinder-hover}"
  icon-button-viewfinder:
    backgroundColor: "{colors.viewfinder-chip}"
    textColor: "{colors.viewfinder-ink}"
    rounded: "{rounded.full}"
    size: "44px"
  button-primary:
    backgroundColor: "{colors.sheet-text}"
    textColor: "{colors.sheet-bg}"
    rounded: "{rounded.radius}"
    padding: "12px 18px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.sheet-text-muted}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.sheet-text}"
    rounded: "{rounded.radius}"
    padding: "12px 18px"
    height: "48px"
  button-quiet-hover:
    backgroundColor: "{colors.sheet-raised}"
  button-small:
    rounded: "{rounded.radius}"
    padding: "6px 12px"
    height: "36px"
  field-input:
    backgroundColor: "{colors.sheet-raised}"
    textColor: "{colors.sheet-text}"
    rounded: "{rounded.radius}"
    padding: "12px 14px"
    height: "48px"
  tools-sheet:
    backgroundColor: "{colors.sheet-bg}"
    textColor: "{colors.sheet-text}"
    rounded: "{rounded.radius-lg}"
  card-summary:
    typography: "{typography.title}"
    padding: "14px 20px"
    height: "64px"
  card-summary-hover:
    backgroundColor: "{colors.sheet-surface}"
  diagnostics-row:
    backgroundColor: "{colors.sheet-raised}"
    rounded: "{rounded.row}"
    padding: "7px 12px"
---

# Design System: Esku

## Overview

**Creative North Star: "The System Viewfinder"**

Esku looks and behaves like the phone's own camera. The viewfinder owns the whole screen at full dynamic height: the live video in a frame, thin corner brackets on the picture, quiet chips for state, the platform's own type set large, and one round shutter disc as the only primary control. Nothing sits on a page of cards; everything either sits on the viewfinder or slides up over it.

The viewfinder follows the system scheme, like the system camera does. In light it is white, with light grey chips and near-black ink; in dark it is black, with graphite chips and white ink. The picture itself is the exception: while the camera is off the video frame is a quiet well in the scheme's own tone, and only while the camera runs does it go black, because the letterbox around a live feed has to recede whatever the page around it does. On white that black is inset and rounded so it reads as a window onto the camera, not a hole; in dark it runs edge to edge. Whatever is drawn over the video (brackets while running, the live guess, the skeleton) uses its own on-video family that never changes with the scheme, because it always sits on moving video over black. The tools are a sheet with the platform's grammar, a rounded top over the camera on a phone and a docked right column on a wide screen, in cool violet-tinted neutrals that also follow the scheme.

Density is low in the viewfinder, where everything is read at arm's length by a second person, and moderate in the sheet, where rows of figures are read up close. The only brand colour on screen is the violet face of the shutter.

**Key Characteristics:**
- Viewfinder and sheet follow the system scheme; the picture is black only while the camera runs.
- Three token families: viewfinder (`--vf-*`, per scheme), on-video (`--ov-*`, fixed), sheet (per scheme).
- Brand violet appears only on the shutter disc.
- The framing brackets carry the tracking state in colour: faint, soft, white, signal yellow.
- Body-part state is a tick or a cross, never colour alone.
- Platform type (system-ui), set large for the live guess and the transcript.
- Pills and circles in the viewfinder; 14px rounded rectangles in the sheet.

## Colors

A scheme-following viewfinder of white or black with one fixed on-video family over the picture and a single yellow signal, beside a scheme-following sheet of violet-tinted neutrals; one violet accent, on one control.

### Primary
- **Shutter Violet** (shutter-violet): the face of the shutter disc and nothing else, in both schemes. Hover lifts to **Lit Shutter** (shutter-violet-hover). Also the colour of the raised-hand mark, which is a brand asset, not a UI colour.

### Secondary
- **Signal Yellow** (signal-yellow): the camera's own "in frame" colour. It paints the framing brackets when hands, face and torso are all framed, over the running video, in both schemes. Never a fill, never text. The dark scheme's focus ring shares its value but is its own token (viewfinder-focus).

### Tertiary
- **Part colours** (part-hands, part-face, part-neck, part-torso, part-arms): identify the five tracked body parts. The same hue draws that part's skeleton on the video and its tick in the chip, so the chip and the drawing can be matched. On the video they are used as is. In the light scheme the chip tick darkens to stay a 3:1 mark on the light chip (`color-mix(in srgb, <part> 55%, viewfinder-ink-light)`). They distinguish parts; they never say whether a part is seen.

### Neutral
- **Viewfinder ground** (viewfinder-bg / viewfinder-bg-light): the viewfinder, the body and `theme-color` (#000000 dark, #ffffff light).
- **Viewfinder ink** (viewfinder-ink / -light): the transcript, the idle title and tagline, the shutter ring and label, the status pill, chip text when a part is seen.
- **Muted ink** (viewfinder-muted / -light): idle notes, the tools button, unseen chips, the cross mark.
- **Soft ink** (viewfinder-soft / -light): the quietest text allowed on the viewfinder: the transcript placeholder.
- **Faint ink** (viewfinder-faint / -light): the brackets while the camera is idle, on the well. Not for text.
- **Hairline** (viewfinder-line / -light): the outline of chip buttons, the transcript scrollbar.
- **Chip** (viewfinder-chip / -light): status pill, part chips, flip button. Graphite in dark, pale grey in light.
- **Well** (viewfinder-well / -light): the video frame while the camera is off. Black in dark, a cool pale grey in light.
- **Focus** (viewfinder-focus / -light): the keyboard focus ring for every viewfinder control; yellow on dark, ink on light.
- **Hover wash** (viewfinder-hover / -light): chip-button hover.
- **Video Black** (video-black): the frame while the camera runs, in both schemes.
- **On-video ink, soft, faint** (on-video-ink, on-video-soft, on-video-faint): the running brackets (hand, searching, nothing yet) and the live-guess text. Fixed white alphas; they never change with the scheme.
- **On-video scrim** (on-video-scrim): behind the live guess, with a 12px backdrop blur.
- **Sheet neutrals, dark / light** (sheet-bg, sheet-surface, sheet-raised, sheet-border, sheet-text, sheet-text-muted, sheet-text-soft and their `-light` pairs): the tools sheet's background and sticky header, hover surface, raised fills (inputs, close button, diagnostic rows), dividers, text, secondary text and the selection tint.
- **Danger** (sheet-danger / sheet-danger-light): one use, a diagnostic block whose numbers are nowhere near training, in weight 600.

### Named Rules
**The One Shutter Rule.** Brand violet lives on the shutter disc and nowhere else. No violet buttons, links, borders or highlights, in either scheme or any family.

**The System Scheme Rule.** The viewfinder and the sheet follow `prefers-color-scheme`: white page, light chips and dark ink in light; black in dark. The only surface that ignores the scheme is the picture while the camera runs, which is always black.

**The On-Video Rule.** Anything drawn over the running video uses the on-video family (`--ov-*`) or signal yellow, never viewfinder tokens, and on-video tokens are never used on the page. Viewfinder tokens (`--vf-*`) are for everything around the picture and on the idle well; sheet tokens are for the tools.

**The Brackets Are the State Rule.** The brackets' colour, inherited from the frame, is the tracking summary in four readable steps: faint on the idle well, soft while searching, white on a hand, signal yellow when framed. Yellow means framed; it is not decoration.

## Typography

**Display Font:** system-ui (with -apple-system, Segoe UI, Roboto, sans-serif)
**Body Font:** the same stack
**Label/Mono Font:** the same stack; figures use tabular numerals.

**Character:** One family, the platform's, so Esku reads as part of the phone's camera. Hierarchy comes from size and weight (400 to 700, with in-between weights 560, 620, 650) rather than from contrast of faces.

### Hierarchy
- **Display** (700, 2rem, 1.1, -0.03em): the "Esku" title on the idle frame only.
- **Live guess** (700, 1.85rem, 1.2, 0.04em, uppercase): the top candidate over the video, single line, ellipsis when long. Read at arm's length.
- **Transcript** (560, 1.4rem, 1.35): the recognised text under the video, three lines visible, then it scrolls; wraps anywhere, never sideways.
- **Headline** (700, 1.15rem, -0.01em): the tools sheet title.
- **Title** (650, 1rem): card summaries; subtitles at 0.95rem 650.
- **Body** (400, 0.92rem, 1.5, max 65ch): card prose in muted text. The idle tagline is 1rem at 1.35, balanced, 24ch.
- **Status** (560, 0.86rem, 1.35): the phase pill; also card notes (0.84rem) and panel status lines (0.86rem, muted).
- **Label** (600, 0.72rem): part chips (also in the compact chip grid), the tools button. The shutter label is 0.86rem 650.
- **Figure** (650, 1.4rem, -0.02em, tabular): the storage figure; diagnostics values are tabular too.

### Named Rules
**The Arm's Length Rule.** What a second person must read (live guess, transcript) is set at 1.4rem or larger in weight 560+, in full ink: viewfinder ink for the transcript, on-video ink on the scrim for the live guess.

## Layout

Phone first. The viewfinder is a flex column at 100dvh (never plain vh): HUD band, then the frame, which is the one element that absorbs spare height, then the caption, then the control bar. The HUD sits above the video, never over the framing: status pill left, flip button right (44px, shown only while running), part chips on a row beneath. The live guess floats over the bottom of the video, centred, 18px up. The control bar is a three-column grid (1fr auto 1fr) so the 72px shutter stays centred at the thumb, with "Herramientas" on the left and an empty balancing cell on the right. Safe-area insets pad the top and bottom.

**The frame.** On a phone in dark the frame runs edge to edge; in light it is inset 12px each side with 22px corners. At 960px and wider it is inset 16px with 22px corners in both schemes. The video uses `object-fit: contain`, and the brackets are placed on the picture's letterbox, 12px in from its corners, never on the frame's box.

**Part chips.** One row, always whole: a second row would eat the video and a scrolled-off chip is a part nobody can read. At 400px and wider they are inline pills (mark beside label, 5px gap). Below 400px they switch to five equal columns (`repeat(5, minmax(0, 1fr))`, 4px gap) with the mark over the label, 12px corners and 0.72rem type, so all five fit whole down to 280px.

**Tools.** On a phone the sheet is fixed to the bottom, 52dvh at most so the top half of the frame stays visible, scrolls internally with contained overscroll, has a sticky header, and closes on Escape, on the close button, or on a tap on the picture. Closed, it is inert. At 960px and wider the viewer becomes a grid (`minmax(0, 1fr) 420px`): the viewfinder is sticky on the left and the sheet becomes an always-open, full-height right column with a 1px left border; the close button disappears and the tools button is hidden but keeps its cell so the shutter stays centred.

Spacing steps are 4, 8, 12, 16, 20 and 24px; card content is inset 20px, the sheet header 20px (24px docked).

## Elevation & Depth

Flat. Depth comes from tone: the well or the black picture against the page, chips a step off the ground, the sheet a step off the viewfinder. The one structural shadow belongs to the phone sheet while it is open over the camera; closed, it casts none, so nothing leaks over the bottom bar. The one blur belongs to the live-guess scrim over moving video. Docked on a wide screen the shadow is removed and a hairline border takes its place.

### Shadow Vocabulary
- **Sheet lift** (`box-shadow: 0 -12px 40px rgba(0, 0, 0, 0.28)`): the tools sheet on a phone, only while open.
- **Scrim blur** (`backdrop-filter: blur(12px)` on on-video-scrim): the live guess only.

### Named Rules
**The Only Lift Rule.** Only the open sheet covering the camera casts a shadow. Everything else is flat or tonal.

## Shapes

Round in the camera, softly square in the sheet. On the viewfinder everything is a pill or a circle (status pill 18px, chips, chip buttons, flip, live guess and shutter all fully round), except the compact chip grid under 400px (12px). The video frame takes 22px corners whenever it is inset (light on a phone, both schemes docked) and is square when it runs edge to edge. The shutter is a 4px inset ring in viewfinder ink around a 56px violet circle that squares down to a 30px, 8px-radius "stop" while running. The framing brackets are 30px L-shapes, 2.5px strokes, with an 8px outer corner. In the sheet, buttons and inputs use 14px, diagnostic rows 10px, and the sheet's top corners 22px; docked, the sheet is square. Take dots are 12px circles. The disclosure chevron is a 9px bordered corner rotated 45 degrees, not a glyph.

### Named Rules
**The Window, Not a Hole Rule.** On white, a black picture is inset and rounded so it reads as a window onto the camera; edge to edge is for the dark scheme on a phone.

## Components

### Buttons
Plain and high-contrast in the sheet; outlined pills on the viewfinder.
- **Shape:** 14px rounded rectangle, 48px minimum height (36px small variant, 6px 12px padding, 0.85rem).
- **Primary:** sheet text colour as fill with sheet background as text (near-white on dark, near-black on light), 12px 18px, weight 600. Hover drops the fill to muted text. One primary per action row.
- **Quiet:** transparent with a sheet-border outline; hover fills with sheet-raised.
- **Disabled:** 50% opacity, not-allowed cursor.
- **Focus:** 2px outline in sheet text, offset 2px.
- **Chip button (viewfinder):** transparent pill with a hairline outline, 44px tall, 0 16px, 0.9rem 600, an 18px line icon before the label; hover washes with viewfinder-hover; focus is a 2px viewfinder-focus ring, offset 3px. Undo and clear only exist once there is text.

### Chips
- **Status pill:** viewfinder chip, viewfinder ink, status type; hidden when empty.
- **Part chip:** viewfinder chip with a 12px mark and a label. Before tracking the mark is a 45% dot in the part colour; seen, it becomes a tick in the part colour (darkened in light) and the label turns to full ink; unseen, a cross in muted ink with a muted label. A visually hidden ": visto" / ": no visto" completes the name for screen readers.

### Cards / Containers
- **Tools sheet:** see Layout; background sheet-bg, 22px top corners, sticky header with a 44px round close button on sheet-raised. No grabber handle: the sheet does not drag, it opens from the tools button and closes by the close button, Escape or a tap on the picture.
- **Card:** a `details` section separated by a 1px sheet-border bottom rule, no fill, no radius. The summary is at least 64px, 14px 20px, title over a muted note (the note carries the key figure so it survives the fold), chevron at the right; hover fills sheet-surface; focus is an inset 2px outline.
- **Diagnostic row:** label left in muted text, tabular value right, on sheet-raised with 10px corners; a wide variant stacks label over value.

### Inputs / Fields
- **Style:** sheet-raised fill, 1px sheet-border, 14px corners, 48px min height, 12px 14px; label above in muted 0.86rem.
- **Focus:** 2px outline in sheet text, offset 1px.

### Navigation
- **Control bar:** the tools button is an icon over a 0.72rem label in muted ink, turning to full ink on hover or while its sheet is open; 64 by 52px minimum; viewfinder-focus ring.

### Video Frame (signature)
The picture. Idle, it is the well (black in dark, pale grey in light) holding the mark, the title, one sentence and the privacy and accuracy lines in viewfinder ink, with the brackets in faint ink. Pressing the shutter turns it black (0.3s colour and background transition) and hands the brackets to the on-video family. The skeleton overlay and the video are mirrored together for the selfie camera and unmirrored for the rear one.

### Shutter Disc (signature)
The one primary control. A 72px disc: 4px ring in viewfinder ink (white on dark, near-black on light), violet face. Pressing it starts reading and the face morphs from circle to rounded square (0.28s, `cubic-bezier(0.16, 1, 0.3, 1)`); the label below switches "Empezar a leer" / "Parar". Disabled while the engine loads, face at 55%. Focus rings the face in viewfinder-focus, offset 3px.

### Live Guess
The top candidate in the live-guess style, on-video ink on the blurred on-video scrim, in both schemes. When a sign lands in the transcript the pill drops 28px and fades (320ms) while the transcript fades up from 55% (420ms). Under reduced motion both are skipped, as are the sheet slide, the frame's colour fade and the shutter morph.

## Do's and Don'ts

### Do:
- **Do** let the viewfinder and the sheet follow the system scheme, and turn the picture black only while the camera runs.
- **Do** use the on-video family (`--ov-*`) and signal yellow for everything drawn over the running video, in both schemes.
- **Do** inset the frame 12px with 22px corners on white; run it edge to edge only in dark on a phone.
- **Do** use shutter violet on the shutter disc face only.
- **Do** show body-part state with a tick or a cross; part colours only tell parts apart and match the skeleton, and the chip tick darkens with `color-mix` in light.
- **Do** let the brackets' colour carry the tracking state: faint idle, soft searching, white hand, signal yellow framed.
- **Do** keep all five part chips on one row, whole; below 400px use five equal columns with the mark over the label.
- **Do** set the live guess and the transcript at 1.4rem or more, weight 560+, in full ink, for reading at arm's length.
- **Do** ring viewfinder controls with viewfinder-focus (yellow on dark, ink on light) and sheet controls with sheet text.
- **Do** size every touch target at 44px or more, and use 100dvh for full-height layout.
- **Do** place anything positioned on the video against the `object-fit: contain` letterbox, and mirror overlays exactly as the video is mirrored.

### Don't:
- **Don't** use violet for buttons, links, focus, borders or highlights anywhere else.
- **Don't** mark "seen" or "not seen" by colour alone.
- **Don't** use signal yellow as a fill or for text; it is the framed brackets only.
- **Don't** use viewfinder tokens over the running video, or on-video tokens on the page.
- **Don't** force the viewfinder dark in the light scheme; only the running picture is black.
- **Don't** wrap the part chips to a second row or let them scroll out of sight.
- **Don't** add shadows beyond the open phone sheet's lift, or blur beyond the live-guess scrim; a closed sheet casts no shadow.
- **Don't** add a grabber handle to the sheet; it does not drag.
- **Don't** cover the top half of the frame with the tools sheet on a phone (52dvh at most).
