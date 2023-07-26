using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EmteqLabs;
using EmteqVR;


// Data saved in : C:\Users\[CurrentUser]\Documents\EmteqLabs\[ApplicationName]\Upload
public class RecordEmteqData : MonoBehaviour
{
    private int participantId = 0;

    public void StartRecord()
    {
        EmteqVRManager.StartRecordingData();
    }

    public void StopRecord()
    {
        EmteqVRManager.StopRecordingData();
    }

    public void StartSection(int videoId)
    {
        EmteqVRManager.StartDataSection(videoId.ToString());
    }

    public void EndSection(int videoId)
    {
        EmteqVRManager.EndDataSection(videoId.ToString());
    }

    public void SetParticipantId(int id)
    {
        participantId = id;
        EmteqVRManager.SetParticipantID(id.ToString());
    }
    
}
