#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>
#include <iostream>
#include <random>
#include <string>

#include "mouse.h"
#include "capture.h"
#include "Arduino.h"
#include "RP2350.h"
#include "sunone_aimbot_2.h"
#include "ghub.h"
#include "rzctl.h"
#include "Teensy41RawHid.h"

namespace
{
aim::AimKalmanSettings buildKalmanSettingsFromConfig()
{
    aim::AimKalmanSettings settings;
    settings.enabled = config.kalman_enabled;
    settings.process_noise_position = static_cast<double>(config.kalman_process_noise_position);
    settings.process_noise_velocity = static_cast<double>(config.kalman_process_noise_velocity);
    settings.measurement_noise = static_cast<double>(config.kalman_measurement_noise);
    settings.velocity_damping = static_cast<double>(config.kalman_velocity_damping);
    settings.max_velocity = static_cast<double>(config.kalman_max_velocity);
    settings.warmup_frames = config.kalman_warmup_frames;
    return settings;
}

double smoothstep(double edge0, double edge1, double value)
{
    if (edge1 <= edge0)
        return value < edge0 ? 0.0 : 1.0;

    const double t = std::clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

double lerp(double a, double b, double t)
{
    return a + (b - a) * t;
}
}

MouseThread::MouseThread(
    int resolution,
    int fovX,
    int fovY,
    double minSpeedMultiplier,
    double maxSpeedMultiplier,
    double predictionInterval,
    bool auto_shoot,
    float bScope_multiplier,
    Arduino* arduinoConnection,
    RP2350* rp2350Connection,
    GhubMouse* gHubMouse,
    KmboxAConnection* Kmbox_A_Connection,
    KmboxNetConnection* Kmbox_Net_Connection,
    MakcuConnection* makcuConnection,
    RzctlMouse* rzctlMouse,
    Teensy41RawHid* teensy41RawHidConnection)
    : screen_width(resolution),
    screen_height(resolution),
    prediction_interval(predictionInterval),
    fov_x(fovX),
    fov_y(fovY),
    max_distance(std::hypot(resolution, resolution) / 2.0),
    min_speed_multiplier(minSpeedMultiplier),
    max_speed_multiplier(maxSpeedMultiplier),
    center_x(resolution / 2.0),
    center_y(resolution / 2.0),
    auto_shoot(auto_shoot),
    bScope_multiplier(bScope_multiplier),
    arduino(arduinoConnection),
    rp2350(rp2350Connection),
    kmbox_a(Kmbox_A_Connection),
    kmbox_net(Kmbox_Net_Connection),
    makcu(makcuConnection),
    gHub(gHubMouse),
    rzctl(rzctlMouse),
    teensy41RawHid(teensy41RawHidConnection),

    prev_velocity_x(0.0),
    prev_velocity_y(0.0),
    prev_x(0.0),
    prev_y(0.0)
{
    prev_time = std::chrono::steady_clock::time_point();
    last_target_time = std::chrono::steady_clock::now();

    wind_mouse_enabled = config.wind_mouse_enabled;
    wind_G = config.wind_G;
    wind_W = config.wind_W;
    wind_M = config.wind_M;
    wind_D = config.wind_D;
    resetWindState();
    clearWindDebugTrail();
    targetKalman.setSettings(buildKalmanSettingsFromConfig());
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;
    lastDetectionDelaySec = 0.0;

    moveWorker = std::thread(&MouseThread::moveWorkerLoop, this);
}

void MouseThread::updateConfig(
    int resolution,
    int fovX,
    int fovY,
    double minSpeedMultiplier,
    double maxSpeedMultiplier,
    double predictionInterval,
    bool auto_shoot,
    float bScope_multiplier
)
{
    screen_width = screen_height = resolution;
    fov_x = fovX;  fov_y = fovY;
    min_speed_multiplier = minSpeedMultiplier;
    max_speed_multiplier = maxSpeedMultiplier;
    prediction_interval = predictionInterval;
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;

    center_x = center_y = resolution / 2.0;
    max_distance = std::hypot(resolution, resolution) / 2.0;

    wind_mouse_enabled = config.wind_mouse_enabled;
    wind_G = config.wind_G; wind_W = config.wind_W;
    wind_M = config.wind_M; wind_D = config.wind_D;
    resetWindState();
    clearWindDebugTrail();
    targetKalman.setSettings(buildKalmanSettingsFromConfig());
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;
    lastDetectionDelaySec = 0.0;
}

MouseThread::~MouseThread()
{
    workerStop = true;
    queueCv.notify_all();
    if (moveWorker.joinable()) moveWorker.join();
}

void MouseThread::queueMove(int dx, int dy)
{
    if (dx == 0 && dy == 0)
    {
        return;
    }

    std::lock_guard lg(queueMtx);
    if (moveQueue.size() >= queueLimit) moveQueue.pop();
    moveQueue.push({ dx,dy });
    queueCv.notify_one();
}

void MouseThread::moveWorkerLoop()
{
    try
    {
        while (!workerStop)
        {
            std::unique_lock ul(queueMtx);
            queueCv.wait(ul, [&] { return workerStop || !moveQueue.empty(); });

            while (!moveQueue.empty())
            {
                Move m = moveQueue.front();
                moveQueue.pop();
                ul.unlock();
                sendMovementToDriver(m.dx, m.dy);
                appendWindDebugStep(m.dx, m.dy);
                ul.lock();
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Mouse] Move worker crashed: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[Mouse] Move worker crashed: unknown exception." << std::endl;
    }
}

void MouseThread::windMouseMoveRelative(int dx, int dy)
{
    if (dx == 0 && dy == 0)
        return;

    windCarryX += static_cast<double>(dx);
    windCarryY += static_cast<double>(dy);

    const double baseG = std::clamp(wind_G, 0.05, 50.0);
    const double baseW = std::clamp(wind_W, 0.0, 80.0);
    const double baseM = std::max(1.0, wind_M);
    const double baseD = std::max(1.0, wind_D);

    std::uniform_real_distribution<double> noiseDist(-1.0, 1.0);
    std::uniform_real_distribution<double> clipDist(0.55, 1.0);
    constexpr double twoPi = 6.28318530717958647692;

    const double carryMag = std::hypot(windCarryX, windCarryY);
    const int maxSubsteps = std::clamp(static_cast<int>(carryMag * 0.24) + 1, 1, 5);

    for (int i = 0; i < maxSubsteps; ++i)
    {
        const double dist = std::hypot(windCarryX, windCarryY);
        const double velMag = std::hypot(windVelX, windVelY);

        if (dist < 0.20 && velMag < 0.12)
            break;

        const double normDist = std::clamp(dist / baseD, 0.0, 1.0);
        const double pullGain = baseG * (0.25 + 0.75 * normDist);
        const double noiseAmp = baseW * (0.15 + 0.85 * normDist);

        double pullX = 0.0;
        double pullY = 0.0;
        if (dist > 1e-8)
        {
            pullX = windCarryX / dist * pullGain;
            pullY = windCarryY / dist * pullGain;
        }

        windPatternRateA = std::clamp(windPatternRateA + noiseDist(windRng) * 0.004, 0.025, 0.280);
        windPatternRateB = std::clamp(windPatternRateB + noiseDist(windRng) * 0.004, 0.025, 0.280);

        const double stepTempo = 0.20 + 0.95 * normDist;
        windPatternPhaseA += windPatternRateA * stepTempo;
        windPatternPhaseB += windPatternRateB * stepTempo;
        if (windPatternPhaseA > twoPi) windPatternPhaseA = std::fmod(windPatternPhaseA, twoPi);
        if (windPatternPhaseB > twoPi) windPatternPhaseB = std::fmod(windPatternPhaseB, twoPi);

        const double oscAX = std::sin(windPatternPhaseA);
        const double oscBX = std::sin(windPatternPhaseB + 1.61803398875);
        const double oscAY = std::cos(windPatternPhaseA * 0.79 + 0.35);
        const double oscBY = std::cos(windPatternPhaseB * 1.17 - 0.48);

        const double patternAmp = baseW * (0.05 + 0.55 * normDist);
        const double patternTargetX = (oscAX + 0.58 * oscBX) * patternAmp;
        const double patternTargetY = (oscAY + 0.58 * oscBY) * patternAmp;
        const double patternBlend = 0.12 + 0.20 * normDist;
        windPatternX = windPatternX * (1.0 - patternBlend) + patternTargetX * patternBlend;
        windPatternY = windPatternY * (1.0 - patternBlend) + patternTargetY * patternBlend;

        windNoiseX = windNoiseX * 0.72 + noiseDist(windRng) * noiseAmp * 0.28;
        windNoiseY = windNoiseY * 0.72 + noiseDist(windRng) * noiseAmp * 0.28;

        const double windForceX = windNoiseX + windPatternX * 0.42;
        const double windForceY = windNoiseY + windPatternY * 0.42;

        const double drag = 0.82 + (1.0 - normDist) * 0.10;
        windVelX = windVelX * drag + pullX + windForceX;
        windVelY = windVelY * drag + pullY + windForceY;

        const double vCap = std::max(0.65, baseM * (0.30 + 0.70 * normDist));
        const double newVelMag = std::hypot(windVelX, windVelY);
        if (newVelMag > vCap)
        {
            const double clip = vCap * clipDist(windRng);
            windVelX = (windVelX / newVelMag) * clip;
            windVelY = (windVelY / newVelMag) * clip;
        }

        windFracX += windVelX;
        windFracY += windVelY;

        int stepX = static_cast<int>(std::round(windFracX));
        int stepY = static_cast<int>(std::round(windFracY));
        if (stepX == 0 && stepY == 0)
            continue;

        windFracX -= static_cast<double>(stepX);
        windFracY -= static_cast<double>(stepY);
        windCarryX -= static_cast<double>(stepX);
        windCarryY -= static_cast<double>(stepY);
        queueMove(stepX, stepY);
    }

    const double carryCap = 120.0;
    const double finalCarryMag = std::hypot(windCarryX, windCarryY);
    if (finalCarryMag > carryCap)
    {
        const double s = carryCap / finalCarryMag;
        windCarryX *= s;
        windCarryY *= s;
    }
}

void MouseThread::resetWindState()
{
    constexpr double twoPi = 6.28318530717958647692;
    std::uniform_real_distribution<double> phaseDist(0.0, twoPi);
    std::uniform_real_distribution<double> rateDist(0.04, 0.16);

    windCarryX = 0.0;
    windCarryY = 0.0;
    windVelX = 0.0;
    windVelY = 0.0;
    windNoiseX = 0.0;
    windNoiseY = 0.0;
    windFracX = 0.0;
    windFracY = 0.0;
    windPatternX = 0.0;
    windPatternY = 0.0;
    windPatternPhaseA = phaseDist(windRng);
    windPatternPhaseB = phaseDist(windRng);
    windPatternRateA = rateDist(windRng);
    windPatternRateB = rateDist(windRng);
}

void MouseThread::appendWindDebugStep(int dx, int dy)
{
    if (dx == 0 && dy == 0)
        return;

    auto delta = mouseCountsToScreenPixels(dx, dy);
    double deltaPxX = delta.first;
    double deltaPxY = delta.second;
    if (std::abs(deltaPxX) < 1e-8 && std::abs(deltaPxY) < 1e-8)
        return;

    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    const auto now = std::chrono::steady_clock::now();
    pruneWindDebugTrailLocked(now);

    if (windDebugTrail.empty())
    {
        windDebugCursorX = center_x;
        windDebugCursorY = center_y;
        windDebugTrail.push_back({ windDebugCursorX, windDebugCursorY, now });
    }

    windDebugCursorX += deltaPxX;
    windDebugCursorY += deltaPxY;
    windDebugTrail.push_back({ windDebugCursorX, windDebugCursorY, now });

    constexpr size_t maxTrailPoints = 220;
    while (windDebugTrail.size() > maxTrailPoints)
        windDebugTrail.pop_front();
}

void MouseThread::pruneWindDebugTrailLocked(const std::chrono::steady_clock::time_point& now)
{
    constexpr auto windTrailLifetime = std::chrono::milliseconds(900);
    while (!windDebugTrail.empty() && (now - windDebugTrail.front().t) > windTrailLifetime)
        windDebugTrail.pop_front();
}

std::pair<double, double> MouseThread::mouseCountsToScreenPixels(int dx, int dy) const
{
    double deltaPxX = static_cast<double>(dx);
    double deltaPxY = static_cast<double>(dy);

    {
        std::lock_guard<std::mutex> cfgLock(configMutex);
        const Config::GameProfile* gpPtr = nullptr;

        auto activeIt = config.game_profiles.find(config.active_game);
        if (activeIt != config.game_profiles.end())
            gpPtr = &activeIt->second;
        else
        {
            auto unifiedIt = config.game_profiles.find("UNIFIED");
            if (unifiedIt != config.game_profiles.end())
                gpPtr = &unifiedIt->second;
        }

        if (gpPtr && gpPtr->sens != 0.0 && gpPtr->yaw != 0.0 && gpPtr->pitch != 0.0)
        {
            const double fovNow = std::max(1.0, fov_x);
            const double fovScale = (gpPtr->fovScaled && gpPtr->baseFOV > 1.0) ? (fovNow / gpPtr->baseFOV) : 1.0;
            const double degX = static_cast<double>(dx) * gpPtr->sens * gpPtr->yaw * fovScale;
            const double degY = static_cast<double>(dy) * gpPtr->sens * gpPtr->pitch * fovScale;

            const double degPerPxX = fov_x / std::max(1.0, screen_width);
            const double degPerPxY = fov_y / std::max(1.0, screen_height);

            if (std::abs(degPerPxX) > 1e-8 && std::abs(degPerPxY) > 1e-8)
            {
                deltaPxX = degX / degPerPxX;
                deltaPxY = degY / degPerPxY;
            }
        }
    }

    return { deltaPxX, deltaPxY };
}

void MouseThread::recordMotionCompensationStep(int dx, int dy)
{
    if (dx == 0 && dy == 0)
        return;

    const auto delta = mouseCountsToScreenPixels(dx, dy);
    if (std::abs(delta.first) < 1e-8 && std::abs(delta.second) < 1e-8)
        return;

    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(motionCompensationMutex);
    pruneMotionCompensationTrailLocked(now);
    motionCompensationTrail.push_back({ delta.first, delta.second, now });
    constexpr size_t maxSamples = 512;
    while (motionCompensationTrail.size() > maxSamples)
        motionCompensationTrail.pop_front();
}

void MouseThread::pruneMotionCompensationTrailLocked(const std::chrono::steady_clock::time_point& now)
{
    constexpr auto motionTrailLifetime = std::chrono::seconds(2);
    while (!motionCompensationTrail.empty() && (now - motionCompensationTrail.front().t) > motionTrailLifetime)
        motionCompensationTrail.pop_front();
}

std::pair<double, double> MouseThread::getMotionCompensationSince(
    std::chrono::steady_clock::time_point since) const
{
    if (since.time_since_epoch().count() == 0)
        return { 0.0, 0.0 };

    double x = 0.0;
    double y = 0.0;
    std::lock_guard<std::mutex> lock(motionCompensationMutex);
    for (const auto& sample : motionCompensationTrail)
    {
        if (sample.t >= since)
        {
            x += sample.x;
            y += sample.y;
        }
    }

    return { x, y };
}

double MouseThread::currentDetectionDelaySec(double observationAgeSec) const
{
    double detectionDelaySec = 0.05;
    if (std::isfinite(observationAgeSec) && observationAgeSec >= 0.0)
    {
        detectionDelaySec = observationAgeSec;
    }
    else
    {
        std::string backend;
        {
            std::lock_guard<std::mutex> lock(configMutex);
            backend = config.backend;
        }
        if (backend == "DML")
        {
            if (dml_detector)
                detectionDelaySec = dml_detector->lastInferenceTimeDML.count() * 0.001;
        }
#ifdef USE_CUDA
        else
        {
            detectionDelaySec = trt_detector.lastInferenceTime.count() * 0.001;
        }
#endif
    }
    if (!std::isfinite(detectionDelaySec))
        detectionDelaySec = 0.05;
    return std::clamp(detectionDelaySec, 0.0, 0.35);
}

double MouseThread::currentPredictionLookaheadSec(double detectionDelaySec) const
{
    double lookahead = std::max(0.0, prediction_interval);
    if (config.kalman_compensate_detection_delay)
        lookahead += std::max(0.0, detectionDelaySec);
    lookahead += static_cast<double>(config.kalman_additional_prediction_ms) * 0.001;
    return std::clamp(lookahead, 0.0, 1.5);
}

std::pair<double, double> MouseThread::predict_target_position(
    double target_x,
    double target_y,
    std::chrono::steady_clock::time_point observationTime)
{
    auto current_time = std::chrono::steady_clock::now();
    if (observationTime.time_since_epoch().count() == 0)
        observationTime = current_time;
    double observationAgeSec = std::chrono::duration<double>(current_time - observationTime).count();
    if (!std::isfinite(observationAgeSec) || observationAgeSec < 0.0)
        observationAgeSec = 0.0;

    targetKalman.setSettings(buildKalmanSettingsFromConfig());

    if (prev_time.time_since_epoch().count() == 0 || !target_detected.load())
    {
        prev_time = observationTime;
        prev_x = target_x;
        prev_y = target_y;
        prev_velocity_x = 0.0;
        prev_velocity_y = 0.0;
        targetKalman.reset();
        const double detectionDelaySec = currentDetectionDelaySec(observationAgeSec);
        const double lookaheadSec = currentPredictionLookaheadSec(detectionDelaySec);
        lastKalmanTelemetry = targetKalman.update(target_x, target_y, 1.0 / 120.0, lookaheadSec);
        lastDetectionDelaySec = detectionDelaySec;
        lastPredictionLookaheadSec = lookaheadSec;
        return { target_x, target_y };
    }

    double dt = std::chrono::duration<double>(observationTime - prev_time).count();
    if (dt < 1e-8) dt = 1e-8;

    prev_time = observationTime;
    prev_x = target_x;
    prev_y = target_y;

    const double detectionDelaySec = currentDetectionDelaySec(observationAgeSec);
    const double lookaheadSec = currentPredictionLookaheadSec(detectionDelaySec);
    lastDetectionDelaySec = detectionDelaySec;
    lastPredictionLookaheadSec = lookaheadSec;

    lastKalmanTelemetry = targetKalman.update(target_x, target_y, dt, lookaheadSec);
    prev_velocity_x = lastKalmanTelemetry.velocity_x;
    prev_velocity_y = lastKalmanTelemetry.velocity_y;

    double predictedX = lastKalmanTelemetry.predicted_x;
    double predictedY = lastKalmanTelemetry.predicted_y;
    if (!std::isfinite(predictedX)) predictedX = target_x;
    if (!std::isfinite(predictedY)) predictedY = target_y;

    return { predictedX, predictedY };
}

void MouseThread::sendMovementToDriver(int dx, int dy)
{
    if (dx == 0 && dy == 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(input_method_mutex);
    const std::string inputMethod = config.input_method;
    bool sent = false;

    if (inputMethod == "TEENSY41_HID")
    {
        if (teensy41RawHid)
            sent = teensy41RawHid->move(dx, dy);
    }
    else if (inputMethod == "RAZER")
    {
        if (rzctl)
            sent = rzctl->mouse_xy(dx, dy);
    }
    else if (inputMethod == "KMBOX_NET")
    {
        if (kmbox_net)
        {
            kmbox_net->move(dx, dy);
            sent = true;
        }
    }
    else if (inputMethod == "KMBOX_A")
    {
        if (kmbox_a)
        {
            kmbox_a->move(dx, dy);
            sent = true;
        }
    }
    else if (inputMethod == "MAKCU")
    {
        if (makcu)
        {
            makcu->move(dx, dy);
            sent = true;
        }
    }
    else if (inputMethod == "RP2350")
    {
        if (rp2350)
        {
            rp2350->move(dx, dy);
            sent = true;
        }
    }
    else if (inputMethod == "ARDUINO")
    {
        if (arduino)
        {
            arduino->move(dx, dy);
            sent = true;
        }
    }
    else if (inputMethod == "GHUB")
    {
        if (gHub)
            sent = gHub->mouse_xy(dx, dy);
    }
    else if (inputMethod == "WIN32")
    {
        INPUT in{ 0 };
        in.type = INPUT_MOUSE;
        in.mi.dx = dx;  in.mi.dy = dy;
        in.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
        sent = (SendInput(1, &in, sizeof(INPUT)) == 1);
    }

    if (sent)
        recordMotionCompensationStep(dx, dy);
}

std::pair<double, double> MouseThread::calc_movement(double tx, double ty)
{
    double offx = tx - center_x;
    double offy = ty - center_y;
    double dist = std::hypot(offx, offy);
    double speed = calculate_speed_multiplier(dist);

    double degPerPxX = fov_x / screen_width;
    double degPerPxY = fov_y / screen_height;

    double mmx = offx * degPerPxX;
    double mmy = offy * degPerPxY;

    double corr = 1.0;
    double fps = static_cast<double>(captureFps.load());
    if (fps > 30.0) corr = 30.0 / fps;

    auto counts_pair = config.degToCounts(mmx, mmy, fov_x);
    double move_x = counts_pair.first * speed * corr;
    double move_y = counts_pair.second * speed * corr;

    return { move_x, move_y };
}

double MouseThread::calculate_speed_multiplier(double distance)
{
    const double snapRadius = std::max(0.0, static_cast<double>(config.snapRadius));
    const double nearRadius = std::max(snapRadius + 1e-6, static_cast<double>(config.nearRadius));
    const double transitionRadius = std::clamp(static_cast<double>(config.closeRangeTransition), 0.0, 80.0);
    const double curveExponent = std::max(0.001, static_cast<double>(config.speedCurveExponent));

    auto nearSpeed = [&](double d)
    {
        double t = std::clamp(d / nearRadius, 0.0, 1.0);
        double curve = 1.0 - std::pow(1.0 - t, curveExponent);
        return min_speed_multiplier +
            (max_speed_multiplier - min_speed_multiplier) * curve;
    };

    auto farSpeed = [&](double d)
    {
        double norm = std::clamp(d / max_distance, 0.0, 1.0);
        return min_speed_multiplier +
            (max_speed_multiplier - min_speed_multiplier) * norm;
    };

    const double snapSpeed = min_speed_multiplier * config.snapBoostFactor;

    if (transitionRadius <= 1e-6)
    {
        if (distance < snapRadius)
            return snapSpeed;

        if (distance < nearRadius)
            return nearSpeed(distance);

        return farSpeed(distance);
    }

    const double snapBlendRadius = std::min(transitionRadius, std::max(0.0, (nearRadius - snapRadius) * 0.45));
    if (snapBlendRadius > 1e-6 && distance < snapRadius + snapBlendRadius)
    {
        const double snapStart = std::max(0.0, snapRadius - snapBlendRadius);
        if (distance <= snapStart)
            return snapSpeed;

        const double t = smoothstep(snapStart, snapRadius + snapBlendRadius, distance);
        return lerp(snapSpeed, nearSpeed(distance), t);
    }

    if (distance < snapRadius)
        return snapSpeed;

    const double nearBlendRadius = std::min(transitionRadius, nearRadius * 0.5);
    const double nearStart = std::max(snapRadius, nearRadius - nearBlendRadius);
    const double nearEnd = nearRadius + nearBlendRadius;
    if (distance < nearStart)
        return nearSpeed(distance);

    if (nearBlendRadius > 1e-6 && distance < nearEnd)
    {
        const double t = smoothstep(nearStart, nearEnd, distance);
        return lerp(nearSpeed(distance), farSpeed(distance), t);
    }

    return farSpeed(distance);
}

bool MouseThread::check_target_in_scope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    double center_target_x = target_x + target_w / 2.0;
    double center_target_y = target_y + target_h / 2.0;

    double reduced_w = target_w * (reduction_factor / 2.0);
    double reduced_h = target_h * (reduction_factor / 2.0);

    double x1 = center_target_x - reduced_w;
    double x2 = center_target_x + reduced_w;
    double y1 = center_target_y - reduced_h;
    double y2 = center_target_y + reduced_h;

    return (center_x > x1 && center_x < x2 && center_y > y1 && center_y < y2);
}

void MouseThread::moveMouse(const AimbotTarget& target)
{
    std::lock_guard lg(input_method_mutex);

    auto predicted = predict_target_position(
        target.x + target.w / 2.0,
        target.y + target.h / 2.0);

    auto mv = calc_movement(predicted.first, predicted.second);
    queueMove(static_cast<int>(mv.first), static_cast<int>(mv.second));
}

void MouseThread::moveMousePivot(
    double pivotX,
    double pivotY,
    std::chrono::steady_clock::time_point observationTime)
{
    std::lock_guard lg(input_method_mutex);
    if (observationTime.time_since_epoch().count() != 0)
    {
        auto cameraDelta = getMotionCompensationSince(observationTime);
        pivotX -= cameraDelta.first;
        pivotY -= cameraDelta.second;
    }
    auto predicted = predict_target_position(pivotX, pivotY, observationTime);
    auto mv = calc_movement(predicted.first, predicted.second);
    int mx = static_cast<int>(mv.first);
    int my = static_cast<int>(mv.second);

    if (mx == 0 && my == 0)
    {
        return;
    }

    if (wind_mouse_enabled)
    {
        windMouseMoveRelative(mx, my);
    }
    else
    {
        queueMove(mx, my);
    }
}

void MouseThread::moveRelative(int dx, int dy)
{
    if (dx == 0 && dy == 0)
        return;
    queueMove(dx, dy);
}

void MouseThread::clearQueuedMoves()
{
    std::lock_guard<std::mutex> lock(queueMtx);
    std::queue<Move> empty;
    moveQueue.swap(empty);
    resetWindState();
}

void MouseThread::pressMouse(const AimbotTarget& target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    bool bScope = check_target_in_scope(target.x, target.y, target.w, target.h, bScope_multiplier);
    if (bScope && !mouse_pressed)
    {
        const std::string inputMethod = config.input_method;
        if (inputMethod == "TEENSY41_HID")
        {
            if (teensy41RawHid)
                teensy41RawHid->press();
        }
        else if (inputMethod == "RAZER")
        {
            if (rzctl)
                rzctl->mouse_down();
        }
        else if (inputMethod == "KMBOX_NET")
        {
            if (kmbox_net)
                kmbox_net->leftDown();
        }
        else if (inputMethod == "KMBOX_A")
        {
            if (kmbox_a)
                kmbox_a->leftDown();
        }
        else if (inputMethod == "MAKCU")
        {
            if (makcu)
                makcu->press(0);
        }
        else if (inputMethod == "RP2350")
        {
            if (rp2350)
                rp2350->press();
        }
        else if (inputMethod == "ARDUINO")
        {
            if (arduino)
                arduino->press();
        }
        else if (inputMethod == "GHUB")
        {
            if (gHub)
                gHub->mouse_down();
        }
        else if (inputMethod == "WIN32")
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = true;
    }
    else if (!bScope && mouse_pressed)
    {
        const std::string inputMethod = config.input_method;
        if (inputMethod == "TEENSY41_HID")
        {
            if (teensy41RawHid)
                teensy41RawHid->release();
        }
        else if (inputMethod == "RAZER")
        {
            if (rzctl)
                rzctl->mouse_up();
        }
        else if (inputMethod == "KMBOX_NET")
        {
            if (kmbox_net)
                kmbox_net->leftUp();
        }
        else if (inputMethod == "KMBOX_A")
        {
            if (kmbox_a)
                kmbox_a->leftUp();
        }
        else if (inputMethod == "MAKCU")
        {
            if (makcu)
                makcu->release(0);
        }
        else if (inputMethod == "RP2350")
        {
            if (rp2350)
                rp2350->release();
        }
        else if (inputMethod == "ARDUINO")
        {
            if (arduino)
                arduino->release();
        }
        else if (inputMethod == "GHUB")
        {
            if (gHub)
                gHub->mouse_up();
        }
        else if (inputMethod == "WIN32")
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = false;
    }
}

void MouseThread::releaseMouse()
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (mouse_pressed)
    {
        const std::string inputMethod = config.input_method;
        if (inputMethod == "TEENSY41_HID")
        {
            if (teensy41RawHid)
                teensy41RawHid->release();
        }
        else if (inputMethod == "RAZER")
        {
            if (rzctl)
                rzctl->mouse_up();
        }
        else if (inputMethod == "KMBOX_NET")
        {
            if (kmbox_net)
                kmbox_net->leftUp();
        }
        else if (inputMethod == "KMBOX_A")
        {
            if (kmbox_a)
                kmbox_a->leftUp();
        }
        else if (inputMethod == "MAKCU")
        {
            if (makcu)
                makcu->release(0);
        }
        else if (inputMethod == "RP2350")
        {
            if (rp2350)
                rp2350->release();
        }
        else if (inputMethod == "ARDUINO")
        {
            if (arduino)
                arduino->release();
        }
        else if (inputMethod == "GHUB")
        {
            if (gHub)
                gHub->mouse_up();
        }
        else if (inputMethod == "WIN32")
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = false;
    }
}

void MouseThread::resetPrediction()
{
    clearQueuedMoves();
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0;
    prev_y = 0;
    prev_velocity_x = 0;
    prev_velocity_y = 0;
    targetKalman.reset();
    lastKalmanTelemetry = {};
    lastPredictionLookaheadSec = 0.0;
    lastDetectionDelaySec = 0.0;
    target_detected.store(false);
}

void MouseThread::checkAndResetPredictions()
{
    auto current_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(current_time - last_target_time).count();
    const double timeoutSec = std::clamp(static_cast<double>(config.kalman_reset_timeout_sec), 0.05, 3.0);

    if (elapsed > timeoutSec && target_detected.load())
    {
        resetPrediction();
    }
}

std::vector<std::pair<double, double>> MouseThread::predictFuturePositions(double pivotX, double pivotY, int frames)
{
    std::vector<std::pair<double, double>> result;
    if (frames <= 0)
        return result;

    result.reserve(frames);

    const double fixedFps = 30.0;
    const double frame_time = 1.0 / fixedFps;

    targetKalman.setSettings(buildKalmanSettingsFromConfig());
    if (targetKalman.initialized())
    {
        const double detectionDelaySec = (lastDetectionDelaySec > 0.0)
            ? lastDetectionDelaySec
            : currentDetectionDelaySec();
        const double baseLookaheadSec = currentPredictionLookaheadSec(detectionDelaySec);
        for (int i = 1; i <= frames; ++i)
        {
            const double t = baseLookaheadSec + frame_time * i;
            auto predicted = targetKalman.predict(t);
            if (!std::isfinite(predicted.first) || !std::isfinite(predicted.second))
                continue;
            result.push_back(predicted);
        }

        if (!result.empty())
            return result;
    }

    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - prev_time).count();

    if (prev_time.time_since_epoch().count() == 0 || dt > 0.5)
    {
        return result;
    }

    double vx = prev_velocity_x;
    double vy = prev_velocity_y;
    
    for (int i = 1; i <= frames; i++)
    {
        double t = frame_time * i;
        double px = pivotX + vx * t;
        double py = pivotY + vy * t;
        result.push_back({ px, py });
    }

    return result;
}

void MouseThread::storeFuturePositions(const std::vector<std::pair<double, double>>& positions)
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions = positions;
}

void MouseThread::clearFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions.clear();
}

std::vector<std::pair<double, double>> MouseThread::getFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    return futurePositions;
}

void MouseThread::clearWindDebugTrail()
{
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    windDebugTrail.clear();
    windDebugCursorX = center_x;
    windDebugCursorY = center_y;
}

std::vector<std::pair<double, double>> MouseThread::getWindDebugTrail()
{
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    const auto now = std::chrono::steady_clock::now();
    pruneWindDebugTrailLocked(now);

    std::vector<std::pair<double, double>> out;
    out.reserve(windDebugTrail.size());
    for (const auto& p : windDebugTrail)
        out.emplace_back(p.x, p.y);
    return out;
}

void MouseThread::setArduinoConnection(Arduino* newArduino)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    arduino = newArduino;
}

void MouseThread::setRP2350Connection(RP2350* newRP2350)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    rp2350 = newRP2350;
}

void MouseThread::setKmboxAConnection(KmboxAConnection* newKmbox_a)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    kmbox_a = newKmbox_a;
}

void MouseThread::setKmboxNetConnection(KmboxNetConnection* newKmbox_net)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    kmbox_net = newKmbox_net;
}

void MouseThread::setMakcuConnection(MakcuConnection* newMakcu)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    makcu = newMakcu;
}

void MouseThread::setGHubMouse(GhubMouse* newGHub)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    gHub = newGHub;
}

void MouseThread::setRzctlMouse(RzctlMouse* newRzctl)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    rzctl = newRzctl;
}

void MouseThread::setTeensy41RawHid(Teensy41RawHid* newTeensy41RawHid)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    teensy41RawHid = newTeensy41RawHid;
}
