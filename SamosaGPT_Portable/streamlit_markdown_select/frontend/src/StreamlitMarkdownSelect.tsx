import React, { useEffect, useState } from 'react';
import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from 'streamlit-component-lib';
import MarkdownSelect from './components/MarkdownSelect.tsx';
import { SelectOption } from './types/MarkdownSelectTypes.ts';
import './index.css';

interface StreamlitArgs {
  options: SelectOption[];
  placeholder: string;
  defaultCodeColor: string;
  maxHeight: string;
  disabled: boolean;
  defaultValue?: string;
}

const StreamlitMarkdownSelect: React.FC<ComponentProps> = ({ args }) => {
  const {
    options = [],
    placeholder = "Select an option...",
    defaultCodeColor = "#00cc88",
    maxHeight = "300px",
    disabled = false,
    defaultValue,
  }: StreamlitArgs = args;

  const [selectedValue, setSelectedValue] = useState<string>(defaultValue || "");

  useEffect(() => {
    Streamlit.setFrameHeight();
    Streamlit.setComponentReady();
  });

  useEffect(() => {
    if (defaultValue !== undefined) {
      setSelectedValue(defaultValue);
    }
  }, [defaultValue]);

  const handleSelectionChange = (value: string) => {
    setSelectedValue(value);
    Streamlit.setComponentValue({
      type: 'selection',
      value: value,
    });
  };

  const handleButtonClick = (optionValue: string, action: string) => {
    const clickData = { optionValue, action };
    
    Streamlit.setComponentValue({
      type: 'button_click',
      value: selectedValue,
      button_data: clickData,
    });
  };

  // Transform options to include button handlers
  const transformedOptions = options.map(option => ({
    ...option,
    buttons: option.buttons?.map(button => ({
      ...button,
      onClick: () => handleButtonClick(option.value, button.action || button.text.toLowerCase()),
    })),
  }));

  return (
    <div style={{ padding: '0', margin: '0' }}>
      <MarkdownSelect
        options={transformedOptions}
        value={selectedValue}
        onChange={handleSelectionChange}
        placeholder={placeholder}
        defaultCodeColor={defaultCodeColor}
        maxHeight={maxHeight}
        disabled={disabled}
        className="w-full"
      />
    </div>
  );
};

const ConnectedStreamlitMarkdownSelect = withStreamlitConnection(StreamlitMarkdownSelect);

export default ConnectedStreamlitMarkdownSelect;