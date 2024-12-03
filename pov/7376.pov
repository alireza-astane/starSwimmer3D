#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 1 }        
    sphere {  m*<0.7861308696045263,0.09076851854363999,9.306574712010908>, 1 }
    sphere {  m*<8.153918067927323,-0.1943237322486222,-5.264102717063023>, 1 }
    sphere {  m*<-6.742045125761663,6.3287576413720235,-3.7732958138814174>, 1}
    sphere { m*<-2.9433990318812895,-5.93068860208811,-1.6126154503731325>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7861308696045263,0.09076851854363999,9.306574712010908>, <-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 0.5 }
    cylinder { m*<8.153918067927323,-0.1943237322486222,-5.264102717063023>, <-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 0.5}
    cylinder { m*<-6.742045125761663,6.3287576413720235,-3.7732958138814174>, <-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 0.5 }
    cylinder {  m*<-2.9433990318812895,-5.93068860208811,-1.6126154503731325>, <-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 1 }        
    sphere {  m*<0.7861308696045263,0.09076851854363999,9.306574712010908>, 1 }
    sphere {  m*<8.153918067927323,-0.1943237322486222,-5.264102717063023>, 1 }
    sphere {  m*<-6.742045125761663,6.3287576413720235,-3.7732958138814174>, 1}
    sphere { m*<-2.9433990318812895,-5.93068860208811,-1.6126154503731325>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7861308696045263,0.09076851854363999,9.306574712010908>, <-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 0.5 }
    cylinder { m*<8.153918067927323,-0.1943237322486222,-5.264102717063023>, <-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 0.5}
    cylinder { m*<-6.742045125761663,6.3287576413720235,-3.7732958138814174>, <-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 0.5 }
    cylinder {  m*<-2.9433990318812895,-5.93068860208811,-1.6126154503731325>, <-0.6330366245956357,-0.8991703953362775,-0.5427153850242414>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    