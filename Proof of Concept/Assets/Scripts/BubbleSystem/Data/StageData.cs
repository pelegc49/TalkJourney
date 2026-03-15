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
        [Tooltip("Ordered display bubbles that form the sentence for this stage.")]
        public List<BubbleData> sentenceBubbles = new List<BubbleData>(); // sentenceBubble that built by DisplayBubbles

        [Tooltip("Optional full sentence audio identifier for single-clip playback mode.")]
        public string fullSentenceAudioIdentifier; // id of the audio or the name of it in the database

        [Header("Selection Bubbles")]
        [Tooltip("Branching options available in this stage.")]
        public List<SelectionBubbleData> selectionBubbles = new List<SelectionBubbleData>();

        [Header("Flow")]
        [Tooltip("If true, this stage can be considered terminal when no selections are present.")]
        public bool canEndJourney; // is it the final bubble? or error bubble? btw, not used in the project..
    }
}
