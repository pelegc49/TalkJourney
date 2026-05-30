using System;
using System.Collections.Generic;
using UnityEngine;

namespace TalkJourney.BubbleSystem.Data
{
    [CreateAssetMenu(
        fileName = "StageData",
        menuName = "TalkJourney/Bubble System/Stage Data",
        order = 0)]
    public class StageData : ScriptableObject
    {
        [Header("Identity")]
        [Tooltip("Stable stage key for diagnostics, analytics, and save/load.")]
        public string stageId;

        [Tooltip("Optional description for designers editing this stage.")]
        [TextArea]
        public string stageNotes;

        [Header("Sentence Bubble")]
        [Tooltip("Localization key for the full sentence. The sentence will be resolved, split into display bubbles, and transliterated at runtime.")]
        public string sentenceLocalizationKey;

        [HideInInspector]
        [Obsolete("Use sentenceLocalizationKey instead.")]
        public List<BubbleData> sentenceBubbles = new List<BubbleData>(); // sentenceBubble that built by DisplayBubbles

        [Header("Selection Bubbles")]
        [Tooltip("Branching options available in this stage.")]
        public List<SelectionBubbleData> selectionBubbles = new List<SelectionBubbleData>();
    }
}
