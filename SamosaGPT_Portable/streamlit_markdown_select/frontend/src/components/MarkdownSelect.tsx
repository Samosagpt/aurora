import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface CodeBlock {
  code: string;
  language?: string;
  color?: string;
}

interface SelectButton {
  text: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
  action?: string;
}

interface SelectOption {
  value: string;
  label: string;
  code_block?: CodeBlock;
  buttons?: SelectButton[];
  disabled?: boolean;
}

interface MarkdownSelectProps {
  options: SelectOption[];
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  defaultCodeColor?: string;
  maxHeight?: string;
  className?: string;
  disabled?: boolean;
}

const MarkdownSelect: React.FC<MarkdownSelectProps> = ({
  options,
  value,
  onChange,
  placeholder = "Select an option...",
  defaultCodeColor = "#00cc88",
  maxHeight = "300px",
  className = "",
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedValue, setSelectedValue] = useState(value || "");
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selectedOption = options.find(opt => opt.value === selectedValue);

  useEffect(() => {
    if (value !== undefined) {
      setSelectedValue(value);
    }
  }, [value]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleOptionSelect = (optionValue: string) => {
    setSelectedValue(optionValue);
    setIsOpen(false);
    onChange?.(optionValue);
  };

  const handleButtonClick = (e: React.MouseEvent, buttonAction: () => void) => {
    e.stopPropagation();
    buttonAction();
  };

  const getButtonVariantStyles = (variant: SelectButton['variant'] = 'primary') => {
    const baseStyles = "px-2 py-1 text-xs rounded transition-colors duration-200 font-medium";
    
    switch (variant) {
      case 'primary':
        return `${baseStyles} bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700`;
      case 'secondary':
        return `${baseStyles} bg-gray-500 text-white hover:bg-gray-600 active:bg-gray-700`;
      case 'danger':
        return `${baseStyles} bg-red-500 text-white hover:bg-red-600 active:bg-red-700`;
      default:
        return `${baseStyles} bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700`;
    }
  };

  const renderCodeBlock = (codeBlock: CodeBlock) => {
    const codeColor = codeBlock.color || defaultCodeColor;
    
    return (
      <div className="ml-auto flex-shrink-0">
        <div 
          className="bg-gray-900 rounded px-2 py-1 text-sm font-mono"
          style={{ color: codeColor }}
        >
          {codeBlock.language && (
            <span className="text-gray-400 text-xs mr-2">{codeBlock.language}</span>
          )}
          <code>{codeBlock.code}</code>
        </div>
      </div>
    );
  };

  const renderButtons = (buttons: SelectButton[]) => {
    return (
      <div className="flex gap-1 ml-2">
        {buttons.map((button, index) => (
          <button
            key={index}
            onClick={(e) => handleButtonClick(e, button.onClick)}
            disabled={button.disabled}
            className={`${getButtonVariantStyles(button.variant)} ${
              button.disabled ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {button.text}
          </button>
        ))}
      </div>
    );
  };

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`
          streamlit-select w-full px-4 py-3 text-left rounded-lg border
          focus:outline-none transition-colors duration-200
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          ${isOpen ? 'ring-2 ring-opacity-50' : ''}
        `}
      >
        <div className="flex items-center justify-between">
          <span className={selectedOption ? '' : 'opacity-60'}>
            {selectedOption?.label || placeholder}
          </span>
          {isOpen ? (
            <ChevronUp className="h-5 w-5 opacity-60" />
          ) : (
            <ChevronDown className="h-5 w-5 opacity-60" />
          )}
        </div>
      </button>

      {isOpen && (
        <div className="streamlit-select-dropdown absolute z-50 w-full mt-1 rounded-lg shadow-lg border">
          <div 
            className="py-1 overflow-y-auto"
            style={{ maxHeight }}
          >
            {options.map((option) => (
              <div
                key={option.value}
                onClick={() => !option.disabled && handleOptionSelect(option.value)}
                className={`
                  streamlit-select-option px-4 py-3 cursor-pointer transition-colors duration-200
                  ${option.disabled 
                    ? 'opacity-50 cursor-not-allowed' 
                    : ''
                  }
                  ${selectedValue === option.value ? 'selected' : ''}
                `}
              >
                <div className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="truncate">{option.label}</span>
                    {option.buttons && renderButtons(option.buttons)}
                  </div>
                  {option.code_block && renderCodeBlock(option.code_block)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default MarkdownSelect;