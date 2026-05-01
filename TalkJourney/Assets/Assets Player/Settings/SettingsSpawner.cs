using UnityEngine;
using UnityEngine.InputSystem;

public class SettingsSpawner : MonoBehaviour
{
    [SerializeField]
    private GameObject settingsPanel;
    [SerializeField]
    private GameObject player;
    [SerializeField]
    private InputActionReference spawnAction;

    private void Start()
    {
        if (settingsPanel != null)
        {
            settingsPanel.SetActive(false);
        }
    }

    private void OnEnable()
    {
        spawnAction.action.performed += SpawnSettingsPanel;
    }

    private void OnDisable()
    {
        spawnAction.action.performed -= SpawnSettingsPanel;
    }

    private void SpawnSettingsPanel(InputAction.CallbackContext context)
    {
        if (settingsPanel != null)
        {
            if (!settingsPanel.activeSelf)
            {
                settingsPanel.transform.position = player.transform.position + player.transform.forward * 2.0f;
                settingsPanel.transform.rotation = Quaternion.LookRotation(player.transform.forward, Vector3.up);
                settingsPanel.SetActive(true);
            }else
            {
                settingsPanel.SetActive(false);
            }
        }
    }
}
