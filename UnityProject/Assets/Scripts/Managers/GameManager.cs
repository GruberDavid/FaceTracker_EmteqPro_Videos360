using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;

public partial class GameManager : MonoBehaviour
{
    public RecordManager recordManager;
    public UIManager uiManager;
    public VideoManager videoManager;

    // Start is called before the first frame update
    void Start()
    {
        uiManager.ShowStartUI();
        videoManager.videoPlayer.loopPointReached += VideoEndReached;
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Backspace))
            VideoSkipped();
    }

    public void SetParticpantId()
    {
        string id = uiManager.inputField.text;
        recordManager.SetParticipantId(Convert.ToInt32(id));
    }

    public void LaunchTrain()
    {
        SetParticpantId();
        uiManager.HideStartUI();
        LaunchPlayUI();
    }

    public void LaunchPlayUI()
    {
        uiManager.HideRatingUI();
        if(recordManager.part1)
            uiManager.ShowPlayUI(videoManager.videoIdx.ToString());
        else
            uiManager.ShowPlayUI((videoManager.videoIdx + 7).ToString());
    }

    public void LaunchWaitUI()
    {
        uiManager.HidePlayUI();
        uiManager.ShowWaitUI();
    }

    public void LaunchVideo()
    {
        uiManager.HideWaitUI();
        videoManager.SetupVideo(true);
        videoManager.PlayCurrentVideo();
        if(videoManager.GetVideoIndex() >= 0)
            recordManager.StartRecording(videoManager.GetVideoIndex());
    }

    public void LaunchRatingUI()
    {
        videoManager.SetupVideo(false);
        uiManager.ShowRatingUI();
    }

    public void LaunchEndScreen()
    {
        uiManager.HideRatingUI();
        uiManager.ShowEndScreen();
    }

    public void NextStep()
    {
        int lastIdx = videoManager.GetVideoIndex();
        videoManager.NextVideo();
        if (videoManager.GetVideoIndex() == -1)
        {
            recordManager.AddLastRating(lastIdx);
            recordManager.SaveRating();
            LaunchEndScreen();
        }
        else
        {
            recordManager.AddLastRating(lastIdx);
            LaunchPlayUI();
        }
    }

    public void CloseApplication()
    {
        Application.Quit();
    }

    private void VideoEndReached(VideoPlayer source)
    {
        recordManager.StopRecording(videoManager.GetVideoIndex());
        LaunchRatingUI();
    }

    public void VideoSkipped()
    {
        if (videoManager.videoPlayer.isPlaying)
        {
            videoManager.videoPlayer.Stop();
            recordManager.StopRecording(videoManager.GetVideoIndex());
            LaunchRatingUI();
        }
    }
}
