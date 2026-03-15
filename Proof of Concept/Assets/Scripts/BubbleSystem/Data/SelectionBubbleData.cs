using System;
using UnityEngine;

namespace TalkJourney.BubbleSystem.Data
{
    [Serializable]
    public class SelectionBubbleData
    {
        [Tooltip("Bubble content and visual configuration for this branching option.")]
        public BubbleData bubble;

        [Tooltip("The stage that should load when this option is clicked or voice-matched.")]
        public StageData nextStage;
    }
}
