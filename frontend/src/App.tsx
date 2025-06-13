import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import FeedScreen from './components/FeedScreen';
import type { Post } from './components/FeedScreen';

export default function App() {
  const [posts, setPosts] = useState<Post[]>([]);

  useEffect(() => {
    const socket = io('http://localhost:8000');
    socket.on('new_post', (post: Post) => {
      setPosts(prev => [post, ...prev]);
    });
    socket.on('update_counts', ({ id, likes, comments }) => {
      setPosts(prev =>
        prev.map(p => p.id === id ? { ...p, likes, comments } : p)
      );
    });
    return () => { socket.disconnect(); };
  }, []);

  const handleLike = (id: string) => {
    // 发送点赞事件给后端
    fetch(`http://localhost:8000/posts/${id}/like`, { method: 'POST' });
  };

  const handleComment = (id: string) => {
    // 简化示例：打开评论输入框或跳转
    console.log('Comment on', id);
  };

  const handleReport = (id: string) => {
    fetch(`http://localhost:8000/posts/${id}/report`, { method: 'POST' });
  };

  return (
    <div className="h-screen flex items-center justify-center bg-gray-50">
      <FeedScreen
        posts={posts}
        onLike={handleLike}
        onComment={handleComment}
        onReport={handleReport}
      />
    </div>
  );
}
