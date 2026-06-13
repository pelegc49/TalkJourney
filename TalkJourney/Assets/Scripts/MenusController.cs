using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class MenusController : MonoBehaviour
{
    [SerializeField]
    private GameObject mainSettings;
    [SerializeField]
    private GameObject audioSettings;
    [SerializeField]
    private GameObject languageSettings;
    [SerializeField]
    private TextMeshProUGUI title;
    [SerializeField]
    private ScrollRect scrollRect;

    private GameObject activeSettings;

    private void Start()
    {
        activeSettings = mainSettings;
        mainSettings.SetActive(true);
        audioSettings.SetActive(false);
        languageSettings.SetActive(false);
        title.text = "General Settings";
        scrollRect.content = mainSettings.GetComponent<RectTransform>();
    }

    public void OpenMainSettings()
    {
        activeSettings.SetActive(false);
        mainSettings.SetActive(true);
        activeSettings = mainSettings;
        title.text = "General Settings";
        scrollRect.content = mainSettings.GetComponent<RectTransform>();
    }

    public void OpenAudioSettings()
    {
        activeSettings.SetActive(false);
        audioSettings.SetActive(true);
        activeSettings = audioSettings;
        title.text = "Audio Settings";
        scrollRect.content = audioSettings.GetComponent<RectTransform>();
    }

    public void OpenLanguageSettings()
    {
        activeSettings.SetActive(false);
        languageSettings.SetActive(true);
        activeSettings = languageSettings;
        title.text = "Language Settings";
        scrollRect.content = languageSettings.GetComponent<RectTransform>();
    }
}
