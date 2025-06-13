import React from 'react';
import { motion } from 'framer-motion';
import { Heart, MessageCircle, AlertTriangle } from 'lucide-react';

export interface Post {
  id: string;
  author: string;
  content: string;
  timestamp: string;
  likes: number;
  comments: number;
}

interface FeedScreenProps {
  posts: Post[];
  onLike: (id: string) => void;
  onComment: (id: string) => void;
  onReport: (id: string) => void;
}

export default function FeedScreen({
  posts,
  onLike,
  onComment,
  onReport,
}: FeedScreenProps) {
  return (
    <div className="max-w-md mx-auto bg-white border border-gray-200 rounded-2xl shadow-lg p-4 space-y-4">
      {/* 标题栏 */}
      <h2 className="text-xl font-semibold text-center mb-2">Simple Twitter Feed</h2>

      {/* 帖子列表 */}
      <div className="space-y-4 overflow-y-auto h-[600px]">
        {posts.map(post => (
          <motion.div
            key={post.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 border rounded-lg"
          >
            {/* 作者与时间 */}
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">{post.author}</span>
              <span className="text-xs text-gray-500">{post.timestamp}</span>
            </div>

            {/* 内容 */}
            <p className="mb-3 break-words">{post.content}</p>

            {/* 操作按钮：点赞、评论、举报 */}
            <div className="flex items-center space-x-6 text-gray-600">
              <button
                onClick={() => onLike(post.id)}
                className="flex items-center space-x-1 hover:text-red-500"
              >
                <Heart size={16} />
                <span>{post.likes}</span>
              </button>
              <button
                onClick={() => onComment(post.id)}
                className="flex items-center space-x-1 hover:text-blue-500"
              >
                <MessageCircle size={16} />
                <span>{post.comments}</span>
              </button>
              <button
                onClick={() => onReport(post.id)}
                className="flex items-center space-x-1 hover:text-yellow-500"
              >
                <AlertTriangle size={16} />
                <span>Report</span>
              </button>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
