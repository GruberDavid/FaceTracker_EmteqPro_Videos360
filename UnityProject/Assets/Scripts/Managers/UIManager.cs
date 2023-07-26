using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System;
using UnityEngine.Video;

public class UIManager : MonoBehaviour
{
    //DesktopUI GameObjects
    public GameObject startUI;
    public TMP_InputField inputField;
    public Button startButton;

    //VRUI GameObjects
    public GameObject playUI;
    public GameObject waitUI;
    public GameObject rateUI;
    public GameObject endScreen;
    public TMP_Text videoNumber;

    // Start is called before the first frame update
    void Start()
    {
        startButton.interactable = false;
    }

    public void SetButtonInteractable()
    {
        startButton.interactable = inputField.text.Length != 0;
    }

    public void ShowStartUI()
    {
        startUI.SetActive(true);
    }

    public void HideStartUI()
    {
        startUI.SetActive(false);
    }

    public void ShowPlayUI(string number)
    {
        //videoNumber.text = number + " / 6";
        videoNumber.text = number + " / 12";
        playUI.SetActive(true);
    }

    public void HidePlayUI()
    {
        playUI.SetActive(false);
    }

    public void ShowWaitUI()
    {
        waitUI.SetActive(true);
    }

    public void HideWaitUI()
    {
        waitUI.SetActive(false);
    }

    public void ShowRatingUI()
    {
        rateUI.SetActive(true);
    }

    public void HideRatingUI()
    {
        rateUI.SetActive(false);
    }

    public void ShowEndScreen()
    {
        endScreen.SetActive(true);
    }

    public void HideEndScreen()
    {
        endScreen.SetActive(false);
    }
}
