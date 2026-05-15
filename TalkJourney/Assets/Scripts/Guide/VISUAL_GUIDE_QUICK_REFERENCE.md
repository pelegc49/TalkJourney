# Visual Guide System - Quick Reference

## Files Created/Modified:
1. ✅ **VisualGuide.cs** (NEW) - Main visual guide behavior script
2. ✅ **GuideController.cs** (MODIFIED) - Added visual guide integration
3. 📋 **VISUAL_GUIDE_SETUP.md** (NEW) - Detailed setup instructions

## What the System Does:

### VisualGuide.cs
- **Show()** → Character appears with "Appear" animation
- **Hide()** → Character disappears with "Disappear" animation  
- **StartTalking()** → Plays audio, triggers "Talk" animation
- **StopTalking()** → Stops audio, returns to idle
- **Auto Update Logic:**
  - Looks at player's camera
  - Moves into view if outside view cone
  - Spatial audio positioned at character

### GuideController.cs Changes
```csharp
public VisualGuide visualGuide; // NEW FIELD - assign in Inspector
```

When `PlayInstruction()` is called:
1. Shows visual guide
2. Displays subtitle text
3. Fetches audio from backend
4. **Plays audio from visual guide's AudioSource** (spatial 3D audio)
5. Hides visual guide when done

---

## Inspector Setup Checklist:

### On Your Guide Character GameObject:
- ✓ Has **VisualGuide** script
- ✓ Has **Animator** (auto-assigned or manual)
- ✓ Has **AudioSource** (auto-created or manual)
  - spatialBlend = 1.0
  - rolloffMode = Logarithmic
- ✓ Animator has these parameters:
  - Trigger: `Appear`
  - Trigger: `Disappear`
  - Bool: `IsTalking`
  - Bool: `IsIdle`

### In GuideController Inspector:
- ✓ **Visual Guide** = Your guide character GameObject
- ✓ **Subtitle Text** = Your subtitle TextMeshPro
- ✓ **Localization Resolver** = Your LocalizationResolver
- ✓ **Audio Backend Client** = Your AudioBackendClient

---

## Key Parameters (Adjust in Inspector):

**VisualGuide component:**
```
Appear Duration:        1.0 sec
Disappear Duration:     1.0 sec
Move Speed:             3.0 units/sec
View Distance:          2.0 units
View Angle Threshold:   30.0 degrees
```

**Animator:**
- All animations should transition properly
- Appear/Disappear animations should match their Duration values above

---

## How It Works (Audio Flow):

```
GuideController.PlayInstruction()
    ↓
visualGuide.Show() → Character appears
    ↓
[Get audio from backend]
    ↓
visualGuide.StartTalking(audioClip) → Audio plays from CHARACTER position (spatial 3D)
    ↓
Character animates, looks at player, moves into view
    ↓
Audio finishes
    ↓
visualGuide.StopTalking()
visualGuide.Hide() → Character disappears with animation
```

---

## Testing (Press Play in Editor):

1. Trigger a guide instruction (Right-click GuideTrigger → Simulate Guide Activation)
2. Character should fade in
3. Subtitle appears
4. Character moves toward you if needed
5. Character talks (mouth animation)
6. Character looks at you
7. Audio pans left/right as you move around character
8. Character fades out when done

---

## Common Tweaks:

| Want This | Change This |
|-----------|------------|
| Faster character movement | Increase `Move Speed` |
| Character appears closer | Decrease `View Distance` |
| Character moves into view more aggressively | Decrease `View Angle Threshold` |
| Smoother animations | Adjust Animator transition times |
| Quieter spatial audio | Lower AudioSource Volume or adjust Max Distance |

---

## Notes:
- Audio is now **3D spatial** - you'll hear it pan based on character position
- Character is **managed by GuideController** - don't manually trigger appearance
- Camera reference comes from **Camera.main** - ensure your main camera is tagged "MainCamera"
- Animations must be in **Animator Controller** before testing

See **VISUAL_GUIDE_SETUP.md** for detailed step-by-step instructions!
