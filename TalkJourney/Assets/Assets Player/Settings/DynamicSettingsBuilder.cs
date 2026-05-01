using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.Events;
using System.Collections.Generic;

public enum SettingUIType
{
    Slider,
    Dropdown,
    Button,
    Toggle,
    SubMenuLink
}

[System.Serializable]
public class SettingItem
{
    public string settingName;
    public SettingUIType uiType;

    [Header("Dropdown Options (Only for Dropdown)")]
    public List<string> dropdownOptions;

    [Header("Slider Settings (Only for Slider)")]
    public float sliderMin = 0f;
    public float sliderMax = 1f;
    public string sliderType = "";

    [Header("Target Sub-Menu (Only for SubMenuLink)")]
    public DynamicSettingsBuilder targetSubMenu;

    [Header("Events")]
    public UnityEvent<float> onSliderValueChanged;
    public UnityEvent<int> onDropdownValueChanged;
    public UnityEvent onButtonClicked;
    public UnityEvent<bool> onToggleValueChanged;
}

public class DynamicSettingsBuilder : MonoBehaviour
{
    [Header("Menu Configuration")]
    public string menuTitle = "Settings";

    [Header("UI Architecture (Assign ONLY on the Main Menu)")]
    public Transform contentArea;
    public TextMeshProUGUI headerTextComponent;

    [Header("XRIT Prefabs")]
    public GameObject sliderPrefab;
    public GameObject dropdownPrefab;
    public GameObject buttonPrefab;
    public GameObject togglePrefab;

    [Header("Settings List")]
    public List<SettingItem> settingsList = new List<SettingItem>();

    private DynamicSettingsBuilder parentMenu;

    private void Start()
    {
        // Only trigger if this is the main menu
        if (contentArea != null)
        {
            OpenMenu(contentArea, headerTextComponent, null);
        }
    }

    public void OpenMenu(Transform targetContentArea, TextMeshProUGUI targetHeader, DynamicSettingsBuilder callerMenu)
    {
        this.contentArea = targetContentArea;
        this.headerTextComponent = targetHeader;
        this.parentMenu = callerMenu;

        // Update the panel title automatically
        if (this.headerTextComponent != null)
        {
            this.headerTextComponent.text = this.menuTitle;
        }

        // Clear existing items in the UI
        foreach (Transform child in contentArea)
        {
            Destroy(child.gameObject);
        }

        // Generate Back button if this is a sub-menu
        if (parentMenu != null)
        {
            CreateBackButton();
        }

        // Generate the UI based on the list in the Inspector
        foreach (SettingItem item in settingsList)
        {
            switch (item.uiType)
            {
                case SettingUIType.Slider:
                    CreateSlider(item);
                    break;
                case SettingUIType.Dropdown:
                    CreateDropdown(item);
                    break;
                case SettingUIType.Button:
                    CreateButton(item);
                    break;
                case SettingUIType.Toggle:
                    CreateToggle(item);
                    break;
                case SettingUIType.SubMenuLink:
                    CreateSubMenuLink(item);
                    break;
            }
        }
    }

    private void CreateBackButton()
    {
        GameObject newObj = Instantiate(buttonPrefab, contentArea);
        SetTitle(newObj, "< Back");
        TextMeshProUGUI titleText = newObj.GetComponentInChildren<TextMeshProUGUI>();
        if (titleText != null)
        {
            titleText.text = "< Back";
        }
        Button button = newObj.GetComponentInChildren<Button>();
        if (button != null)
        {
            button.onClick.AddListener(() =>
            {
                parentMenu.OpenMenu(this.contentArea, this.headerTextComponent, parentMenu.parentMenu);
            });
        }
    }

    private void CreateSubMenuLink(SettingItem item)
    {
        GameObject newObj = Instantiate(buttonPrefab, contentArea);
        SetTitle(newObj, item.settingName + " >");

        Button button = newObj.GetComponentInChildren<Button>();
        if (button != null)
        {
            button.onClick.AddListener(() =>
            {
                if (item.targetSubMenu != null)
                {
                    // Pass the shared content area and header text to the next menu
                    item.targetSubMenu.OpenMenu(this.contentArea, this.headerTextComponent, this);
                }
            });
        }
    }

    private void CreateSlider(SettingItem item)
    {
        GameObject newObj = Instantiate(sliderPrefab, contentArea);
        SetTitle(newObj, item.settingName);

        Slider slider = newObj.GetComponentInChildren<Slider>();

        if (slider != null)
        {
            slider.minValue = item.sliderMin;
            slider.maxValue = item.sliderMax;

            TextMeshProUGUI valueText = GetValueTextComponent(newObj);
            TextMeshProUGUI typeText = GetTypeTextComponent(newObj);

            if (typeText != null)
            {
                typeText.text = item.sliderType;
            }

            // Set the initial text value before the user interacts
            if (valueText != null)
            {
                valueText.text = slider.value.ToString("0.0");
            }

            // Listen to slider changes
            slider.onValueChanged.AddListener((val) =>
            {
                // Automatically update the UI text
                if (valueText != null)
                {
                    valueText.text = val.ToString("0.0");
                }

                // Invoke the custom event from the inspector
                item.onSliderValueChanged.Invoke(val);
            });
        }
    }

    private TextMeshProUGUI GetValueTextComponent(GameObject parentObj)
    {
        Transform[] allChildren = parentObj.GetComponentsInChildren<Transform>(true);
        foreach (Transform child in allChildren)
        {
            if (child.name == "Value Text")
            {
                return child.GetComponent<TextMeshProUGUI>();
            }
        }
        return null;
    }

    private TextMeshProUGUI GetTypeTextComponent(GameObject parentObj)
    {
        Transform[] allChildren = parentObj.GetComponentsInChildren<Transform>(true);
        foreach (Transform child in allChildren)
        {
            if (child.name == "Type Text")
            {
                return child.GetComponent<TextMeshProUGUI>();
            }
        }
        return null;
    }

    private void CreateDropdown(SettingItem item)
    {
        GameObject newObj = Instantiate(dropdownPrefab, contentArea);
        SetTitle(newObj, item.settingName);

        TMP_Dropdown dropdown = newObj.GetComponentInChildren<TMP_Dropdown>();
        if (dropdown != null)
        {
            dropdown.ClearOptions();
            dropdown.AddOptions(item.dropdownOptions);
            dropdown.onValueChanged.AddListener((val) => item.onDropdownValueChanged.Invoke(val));
        }
    }

    private void CreateButton(SettingItem item)
    {
        GameObject newObj = Instantiate(buttonPrefab, contentArea);
        SetTitle(newObj, item.settingName);

        Button button = newObj.GetComponentInChildren<Button>();
        if (button != null)
        {
            button.onClick.AddListener(() => item.onButtonClicked.Invoke());
        }
    }

    private void CreateToggle(SettingItem item)
    {
        GameObject newObj = Instantiate(togglePrefab, contentArea);
        SetTitle(newObj, item.settingName);

        Toggle toggle = newObj.GetComponentInChildren<Toggle>();
        if (toggle != null)
        {
            toggle.onValueChanged.AddListener((val) => item.onToggleValueChanged.Invoke(val));
        }
    }

    private void SetTitle(GameObject obj, string title)
    {
        // Find the Label object in the XRIT prefabs to set the text
        Transform labelTransform = obj.transform.Find("Label");
        if (labelTransform != null)
        {
            TextMeshProUGUI titleText = labelTransform.GetComponent<TextMeshProUGUI>();
            if (titleText != null)
            {
                titleText.text = title;
            }
        }
    }
}