// 设备和浏览器检测JavaScript
// Device and browser detection JavaScript
function detectDeviceAndBrowser() {
    const userAgent = navigator.userAgent.toLowerCase();
    let device = 'desktop';
    let browser = 'other';
    
    // 设备检测
    // Device detection
    if (userAgent.match(/android|webos|iphone|ipad|ipod|blackberry|windows phone/i)) {
        device = 'mobile';
        document.body.classList.add('mobile-device');
    } else {
        document.body.classList.add('desktop-device');
    }
    
    // 浏览器检测
    // Browser detection
    if (userAgent.indexOf('chrome') > -1 && userAgent.indexOf('edge') === -1) {
        browser = 'chrome';
        document.body.classList.add('chrome-browser');
    } else if (userAgent.indexOf('firefox') > -1) {
        browser = 'firefox';
        document.body.classList.add('firefox-browser');
    } else if (userAgent.indexOf('safari') > -1 && userAgent.indexOf('chrome') === -1) {
        browser = 'safari';
        document.body.classList.add('safari-browser');
    } else {
        document.body.classList.add('other-browser');
    }
    
    // 根据设备和浏览器应用CSS
    // Apply CSS based on device and browser
    applyResponsiveStyles(device, browser);
    
    console.log(`检测到: ${device} 设备，${browser} 浏览器`);
    return {device, browser};
}

function applyResponsiveStyles(device, browser) {
    // 根据设备应用CSS
    // Apply CSS based on device
    let deviceCss = document.createElement('link');
    deviceCss.rel = 'stylesheet';
    deviceCss.href = device === 'mobile' ? '/static/css/mobile.css' : '/static/css/desktop.css';
    document.head.appendChild(deviceCss);
    
    // 根据浏览器应用CSS
    // Apply CSS based on browser
    let browserCss = document.createElement('link');
    browserCss.rel = 'stylesheet';
    browserCss.href = `/static/css/${browser}.css`;
    document.head.appendChild(browserCss);
}

// 页面加载时运行检测
// Run detection on page load
window.addEventListener('DOMContentLoaded', detectDeviceAndBrowser);
