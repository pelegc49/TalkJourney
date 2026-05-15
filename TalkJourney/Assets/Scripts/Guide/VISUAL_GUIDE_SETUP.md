# Visual Guide System - Unity Editor Setup Guide

This guide walks you through setting up the visual guide character system in Unity.

## Overview

The visual guide system consists of:
- **VisualGuide.cs** - Script attached to your guide character that manages animations, positioning, and spatial audio
- **GuideController.cs** (modified) - Now controls when the visual guide appears/disappears
- **Animator Controller** - Your existing animations wired to specific parameters

---

## Step 1: Prepare Your Guide Character Model

1. In your scene or prefab folder, locate your guide character model
2. If you're using a prefab, open it for editing
3. **Important**: The character should start **disabled** (uncheck the checkbox in the Inspector)
4. Make sure your character has an **Animator** component attached
5. Your Animator should have animations for:
   - `IdleOff` (hidden state - scale 0)
   - `Appear` (fade in or scale up animation)
   - `Idle` (visible idle - neutral standing pose)
   - `Talk` (mouth moving, lip sync animation)
   - `Disappear` (fade out or scale down animation)

---

## Step 2: Add the VisualGuide Script

1. In the Inspector, find your guide character GameObject
2. Click **Add Component**
3. Search for and add **VisualGuide** script
4. The script will auto-find the **Animator** and **AudioSource** components if they exist
5. If not auto-found, manually assign them:
   - **Animator**: Drag your character's Animator component
   - **AudioSource**: Create one if it doesn't exist (Add Component > Audio > Audio Source)

---

## Step 3: Configure the AudioSource for Spatial 3D Audio

The AudioSource will be automatically configured by VisualGuide, but verify:

1. Select the guide character
2. In the Inspector, find the **AudioSource** component
3. Settings:
   - **Spatial Blend**: `1.0` (fully 3D) - ✓ Auto-set by VisualGuide
   - **Rolloff Mode**: `Logarithmic` - ✓ Auto-set by VisualGuide
   - **Volume Rolloff**: Leave at default (controls how sound fades with distance)
   - **Doppler Level**: `1.0` - ✓ Auto-set by VisualGuide
   - **Volume**: `1.0`

---

## Step 4: Configure VisualGuide Inspector Parameters

In the Inspector, with your guide character selected, find the **VisualGuide** component and configure:

### Appearance
- **Appear Duration**: `1.0` (seconds for appear animation)
- **Disappear Duration**: `1.0` (adjust based on your animation length)

### Movement
- **Move Speed**: `3.0` (units per second - how fast guide moves into view)
- **View Distance**: `2.0` (distance in front of player where guide appears)
- **View Angle Threshold**: `30.0` (degrees - guide considered "in view" if within this cone)

### Animation Parameters
These must match your Animator's parameter names exactly:
### Animation Parameters
These must match your Animator's parameter names exactly:
- **Visibility Parameter**: `IsVisible` (boolean parameter in Animator)
- **Talking Parameter**: `IsTalking` (boolean parameter in Animator)

---

## Step 5: Set Up Animator Controller

Your Animator Controller needs these parameters and transitions:

### Required Animator Parameters:
1. **Booleans** (right-click Parameters panel → Add Parameter → Bool):
   - `IsVisible` (controls IdleOff → Appear → Idle → Disappear → IdleOff)
   - `IsTalking`

### Animation States & Transitions:

Create these states in your Animator (if not already present):
- **IdleOff** (default state — scale 0)
- **Appear** (entrance animation)
- **Idle** (visible idle — scale 1)
- **Talk** (talking/lip-sync animation)
- **Disappear** (exit animation)

### Transitions to Create:

1. **IdleOff** (default) → **Appear**
   - Condition: `IsVisible` = true
   - Leave "Has Exit Time" unchecked on this outgoing transition; the Appear state itself should have exit time so it proceeds to Idle automatically.

2. **Appear** → **Idle**
   - Condition: none
   - Check "Has Exit Time" on Appear so it finishes into Idle

3. **Idle** → **Talk**
   - Condition: `IsTalking` = true
   - Uncheck "Has Exit Time"

4. **Talk** → **Idle**
   - Condition: `IsTalking` = false
   - Uncheck "Has Exit Time"

5. **Idle** → **Disappear**
   - Condition: `IsVisible` = false
   - Let Disappear play (use Has Exit Time on Disappear if you want it to finish before moving)

6. **Disappear** → **IdleOff**
   - Condition: none
   - Use "Has Exit Time" on Disappear so it naturally returns to IdleOff when finished

---

## Step 6: Wire Up GuideController

1. Find your **GuideController** in the scene
2. In the Inspector, find the **Guide Controller** component
3. **Drag** your guide character GameObject into the **Visual Guide** field
   - This should be the same GameObject with the VisualGuide script

4. Verify these are still assigned:
   - **Audio Source**: (optional fallback if no visual guide)
   - **Subtitle Text**: Your subtitle TextMeshPro component
   - **Localization Resolver**: Your LocalizationResolver instance
   - **Audio Backend Client**: Your AudioBackendClient instance

---

## Step 7: Animation Timing (Important!)

Make sure your animations have reasonable lengths:

1. In your Animator, select each animation clip:
   - **Appear animation**: Should be ~0.5 - 1.5 seconds
   - **Disappear animation**: Should be ~0.5 - 1.5 seconds
   - **Talk animation**: Can be a loop or ~2-3 seconds (will repeat while talking)
   - **Idle animation**: Can loop indefinitely

2. Update the VisualGuide script's durations to match:
   - **Appear Duration** = your Appear animation length
   - **Disappear Duration** = your Disappear animation length

---

## Step 8: Test the Setup

### In Editor:
1. Press **Play**
2. Trigger a guide instruction (e.g., use Context Menu on GuideTrigger → "Simulate Guide Activation")
3. Verify:
   - ✓ Character appears with animation
   - ✓ Character talks and moves toward your view
   - ✓ Subtitle text appears below/near the character
   - ✓ Audio plays from character's position (you should hear it pan left/right if you move)
   - ✓ Character looks at you while talking
   - ✓ Character disappears with animation when done

### Common Issues:

**Problem**: Character doesn't appear
- **Solution**: Check that the GameObject is set in GuideController's Visual Guide field

**Problem**: Character appears but doesn't animate
- **Solution**: Verify Animator parameters match VisualGuide script exactly (case-sensitive)

**Problem**: Audio doesn't move with character
- **Solution**: Verify AudioSource.spatialBlend = 1.0 (fully 3D)

**Problem**: Character not looking at player
- **Solution**: Check that the character's forward direction (+Z axis) points "forward" in your model

**Problem**: Character doesn't move into view
- **Solution**: Verify View Angle Threshold is reasonable (default 30°)

---

## Step 9: Fine-Tune Behavior

Once basic functionality works, adjust:

1. **Move Speed**: Increase for faster movement into view, decrease for slower
2. **View Distance**: Closer = character appears nearer to player, farther = appears further back
3. **View Angle Threshold**: Lower = character moves into view more aggressively, higher = more lenient
4. **Animation Transitions**: Adjust transition times for smoother blending between animations

---

## Architecture Summary

**Flow when guide instruction is triggered:**

1. GuideController calls `visualGuide.Show(Camera.main.transform)`
2. VisualGuide appears with Appear animation
3. Subtitle text displays
4. Audio clip fetched from backend
5. VisualGuide.StartTalking(clip) → plays audio, triggers Talk animation
6. Character moves into player's view if needed
7. Character looks at player
8. While audio plays, character stays visible and talking
9. When audio ends, GuideController calls `visualGuide.StopTalking()` then `visualGuide.Hide()`
10. VisualGuide plays Disappear animation
11. Audio comes from character's position (3D spatial audio) - pans based on character's relative position

---

## Optional: Customize Further

**To adjust character movement behavior**, in VisualGuide.cs `MoveToPlayerView()` method, you can:
- Change the target position (currently: `cameraPosition + cameraForward * viewDistance`)
- Add easing/deceleration
- Add constraints (min/max distance, height bounds)

**To customize appearance/disappearance**, you can:
- Adjust the animation durations to match your exact animation clip lengths
- Use the animation's Exit Time for auto-transition instead of triggers

---

## Checklist

- [ ] Guide character model imported and ready
- [ ] VisualGuide.cs script added to character
- [ ] AudioSource configured for spatial 3D (spatialBlend = 1.0)
- [ ] Animator Controller has Appear/Disappear/Talk animations
- [ ] Animator Controller has Appear/Disappear/Talk/IdleOff animations
- [ ] Animator Parameters created (`IsVisible`, `IsTalking` booleans)
- [ ] Animator Transitions set up correctly
- [ ] GuideController's Visual Guide field populated
- [ ] VisualGuide parameters adjusted to match animation times
- [ ] Tested in editor - character appears/moves/talks/disappears
- [ ] Audio heard from character's position (spatial)

---

## Done! 🎉

Your visual guide system is now ready. The character will automatically appear when guide instructions play, follow the player's view, animate with talking motion, and disappear when finished. Audio will be spatial and come from the character's world position.
