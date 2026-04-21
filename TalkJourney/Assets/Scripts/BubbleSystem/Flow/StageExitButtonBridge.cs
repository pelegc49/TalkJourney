using UnityEngine;

namespace TalkJourney.BubbleSystem.Flow
{
    public class StageExitButtonBridge : MonoBehaviour
    {
        public void ExitActiveStage()
        {
            var activeEntrance = StageEntrance.ActiveStageEntrance;
            if (activeEntrance == null)
            {
                Debug.LogWarning("StageExitButtonBridge could not find an active StageEntrance.", this);
                return;
            }

            activeEntrance.ExitStage();
        }
    }
}
