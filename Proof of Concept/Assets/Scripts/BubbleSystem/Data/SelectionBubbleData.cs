using System;
using UnityEngine;

namespace TalkJourney.BubbleSystem.Data
{
    [Serializable]
    public class SelectionBubbleData
    {
        [Tooltip("Bubble content and visual configuration for this branching option.")]
        public BubbleData bubble;

        [Tooltip("Marks this option as the only correct selection for the stage.")]
        public bool isCorrect;

        [Tooltip("The stage that should load when this option is clicked or voice-matched.")]
        public StageData nextStage;
    }
}
