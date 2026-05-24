#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <vector>

#include "capture.h"
#include "mouse.h"
#include "sunone_aimbot_2.h"
#include "runtime/thread_loops.h"

void createInputDevices();
void assignInputDevices();
void handleEasyNoRecoil(MouseThread& mouseThread)
{
    bool easyNoRecoil = false;
    int recoil_compensation = 0;
    {
        std::lock_guard<std::mutex> cfgLock(configMutex);
        easyNoRecoil = config.easynorecoil;
        recoil_compensation = static_cast<int>(config.easynorecoilstrength);
    }

    if (easyNoRecoil && shooting.load() && zooming.load())
    {
        mouseThread.moveRelative(0, recoil_compensation);
    }
}

void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastVersion = -1;
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    MultiTargetTracker targetTracker;
    std::optional<AimbotTarget> activeTarget;
    int activeTrackId = -1;
    bool activeTargetObserved = false;
    bool wasAiming = false;
    auto lastTrackerUpdate = std::chrono::steady_clock::time_point::min();

    auto resetActiveTarget = [&]() {
        activeTarget.reset();
        activeTrackId = -1;
        activeTargetObserved = false;
        mouseThread.clearFuturePositions();
        mouseThread.resetPrediction();
    };

    while (!shouldExit)
    {
        bool hasNewDetection = false;
        bool hasAimObservation = false;
        int detectionResolution = 0;
        bool disableHeadshot = false;
        int predictionFuturePositions = 0;
        bool autoShoot = false;

        {
            std::lock_guard<std::mutex> cfgLock(configMutex);
            detectionResolution = config.detection_resolution;
            disableHeadshot = config.disable_headshot;
            predictionFuturePositions = config.prediction_futurePositions;
            autoShoot = config.auto_shoot;
        }

        const bool aimingNow = aiming.load();
        if (aimingNow != wasAiming)
        {
            resetActiveTarget();
            wasAiming = aimingNow;
        }

        {
            std::unique_lock<std::mutex> lock(detectionBuffer.mutex);
            detectionBuffer.cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                return detectionBuffer.version > lastVersion || shouldExit;
                }
            );

            if (shouldExit) break;

            if (detectionBuffer.version > lastVersion)
            {
                boxes = detectionBuffer.boxes;
                classes = detectionBuffer.classes;
                lastVersion = detectionBuffer.version;
                hasNewDetection = true;
            }
        }

        if (input_method_changed.load())
        {
            createInputDevices();
            assignInputDevices();
            input_method_changed.store(false);
        }

        if (detection_resolution_changed.load())
        {
            {
                std::lock_guard<std::mutex> cfgLock(configMutex);
                mouseThread.updateConfig(
                    config.detection_resolution,
                    config.fovX,
                    config.fovY,
                    config.minSpeedMultiplier,
                    config.maxSpeedMultiplier,
                    config.predictionInterval,
                    config.auto_shoot,
                    config.bScope_multiplier
                );
            }
            targetTracker.reset();
            {
                std::lock_guard<std::mutex> lk(g_trackerDebugMutex);
                g_trackerDebugTracks.clear();
                g_trackerLockedId = -1;
            }
            resetActiveTarget();
            detection_resolution_changed.store(false);
        }

        if (hasNewDetection)
        {
            targetTracker.update(
                boxes,
                classes,
                detectionResolution,
                detectionResolution,
                disableHeadshot,
                aimingNow
            );
            lastTrackerUpdate = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lk(g_trackerDebugMutex);
                g_trackerDebugTracks = targetTracker.getDebugTracks();
                g_trackerLockedId = targetTracker.getLockedTrackId();
            }

            LockedTargetInfo lockInfo;
            if (targetTracker.getLockedTarget(lockInfo))
            {
                if (activeTrackId != -1 && activeTrackId != lockInfo.trackId)
                {
                    mouseThread.resetPrediction();
                    mouseThread.clearFuturePositions();
                }

                activeTarget = lockInfo.target;
                activeTrackId = lockInfo.trackId;
                activeTargetObserved = lockInfo.observedThisFrame;
                mouseThread.setTargetDetected(true);

                if (lockInfo.observedThisFrame)
                {
                    hasAimObservation = true;
                    mouseThread.setLastTargetTime(std::chrono::steady_clock::now());

                    auto futurePositions = mouseThread.predictFuturePositions(
                        activeTarget->pivotX,
                        activeTarget->pivotY,
                        predictionFuturePositions
                    );
                    mouseThread.storeFuturePositions(futurePositions);
                }
            }
            else
            {
                resetActiveTarget();
            }
        }

        if (activeTarget)
        {
            const int fps = std::max(1, captureFps.load());
            const int staleMs = std::clamp(2000 / fps, 25, 180);
            if (std::chrono::steady_clock::now() - lastTrackerUpdate > std::chrono::milliseconds(staleMs))
            {
                resetActiveTarget();
            }
        }

        if (aimingNow)
        {
            if (activeTarget && hasAimObservation)
            {
                mouseThread.moveMousePivot(activeTarget->pivotX, activeTarget->pivotY);

                if (autoShoot)
                {
                    mouseThread.pressMouse(*activeTarget);
                }
            }
            else
            {
                if (!activeTarget || !activeTargetObserved)
                {
                    mouseThread.clearQueuedMoves();
                }

                if (autoShoot)
                {
                    mouseThread.releaseMouse();
                }
            }
        }
        else
        {
            mouseThread.clearQueuedMoves();
            if (autoShoot)
            {
                mouseThread.releaseMouse();
            }
        }

        handleEasyNoRecoil(mouseThread);

        mouseThread.checkAndResetPredictions();
    }
}


