/* Follow appended output only while the reader is at its bottom. */
globalThis.BabyAILogFollow = function createFollowingOutput(element) {
  let following = true;
  element.addEventListener('scroll', () => {
    following = element.scrollHeight - element.clientHeight - element.scrollTop <= 48;
  });
  return {
    reset() { following = true; },
    setText(text) {
      if (element.textContent === text) return;
      const top = element.scrollTop;
      element.textContent = text;
      element.scrollTop = following ? element.scrollHeight : top;
    }
  };
};
