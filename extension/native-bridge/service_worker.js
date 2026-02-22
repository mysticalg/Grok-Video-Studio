const HOST_NAME = 'com.grok.video_studio.host';

function sendNative(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendNativeMessage(HOST_NAME, message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      resolve(response);
    });
  });
}

chrome.runtime.onInstalled.addListener(async () => {
  try {
    const ping = await sendNative({ id: 'install-ping', action: 'ping' });
    console.log('Native host ping response:', ping);
  } catch (err) {
    console.warn('Native host unavailable during install:', err);
  }
});

chrome.action.onClicked.addListener(async () => {
  try {
    const result = await sendNative({
      id: `click-${Date.now()}`,
      action: 'social_upload_step',
      payload: {
        platform: 'youtube',
        step: 'status_probe'
      }
    });
    console.log('social_upload_step result:', result);
  } catch (err) {
    console.error('Native bridge error:', err);
  }
});
