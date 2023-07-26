using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using UnityEngine;
using UnityEngine.Video;

public class VideoManager : MonoBehaviour
{
    // True : All videos are shuffled ; False : All videos except the first are shuffled
    public bool allRandom;

    public List<VideoClip> videos;
    public VideoPlayer videoPlayer;

    public int videoIdx = 0;
    private List<int> indexes;

    public RenderTexture videoRenderTexture;
    public Material defaultSkyboxMaterial;
    public Material videoMaterial;

    // Start is called before the first frame update
    void Start()
    {
        indexes = new List<int>();
        if (allRandom)
        {
            List<int> tmp = new List<int>(Enumerable.Range(0, videos.Count));
            tmp.Shuffle();
            indexes.AddRange(tmp);
            foreach (int i in indexes)
                Debug.Log(i);
        }
        else
        {
            indexes.Add(0);
            List<int> tmp = new List<int>(Enumerable.Range(1, videos.Count - 1));
            tmp.Shuffle();
            indexes.AddRange(tmp);
            foreach (int i in indexes)
                Debug.Log(i);
        }

        defaultSkyboxMaterial = RenderSettings.skybox;
        videoMaterial.mainTexture = videoRenderTexture;
    }

    public VideoClip GetCurrentVideo()
    {
        if (videoIdx != -1)
        {
            return videos[indexes[videoIdx]];
        }
        else
        {
            return null;
        }
    }

    public int GetVideoIndex()
    {
        if (videoIdx != -1)
            return indexes[videoIdx];
        else
            return -1;
    }

    public void NextVideo()
    {
        videoIdx++;
        if (videoIdx >= videos.Count)
            videoIdx = -1;
    }

    public void PlayCurrentVideo()
    {
        if (videoPlayer.clip != null)
        {
            videoPlayer.Play();
        }
    }

    public void SetupVideo(bool v)
    {
        if (v)
        {
            VideoClip video = GetCurrentVideo();
            videoPlayer.clip = video;

            RenderTexture videoRT = new RenderTexture((int)video.width, (int)video.height, 0, RenderTextureFormat.Default, RenderTextureReadWrite.Default);

            videoMaterial.mainTexture = videoRT;
            videoPlayer.targetTexture = videoRT;

            RenderSettings.skybox = videoMaterial;
        }
        else
        {
            RenderSettings.skybox = defaultSkyboxMaterial;
        }
    }
}
