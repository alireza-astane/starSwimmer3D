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
    sphere { m*<0.8781810153327281,0.6630532524673605,0.3851066499699086>, 1 }        
    sphere {  m*<1.121631880034474,0.7196744488307629,3.374673823477221>, 1 }
    sphere {  m*<3.6148790690970105,0.7196744488307627,-0.8426083850133965>, 1 }
    sphere {  m*<-2.445355063418057,5.760952913218832,-1.579975969539605>, 1}
    sphere { m*<-3.858763143237336,-7.686285202753245,-2.415009660965361>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.121631880034474,0.7196744488307629,3.374673823477221>, <0.8781810153327281,0.6630532524673605,0.3851066499699086>, 0.5 }
    cylinder { m*<3.6148790690970105,0.7196744488307627,-0.8426083850133965>, <0.8781810153327281,0.6630532524673605,0.3851066499699086>, 0.5}
    cylinder { m*<-2.445355063418057,5.760952913218832,-1.579975969539605>, <0.8781810153327281,0.6630532524673605,0.3851066499699086>, 0.5 }
    cylinder {  m*<-3.858763143237336,-7.686285202753245,-2.415009660965361>, <0.8781810153327281,0.6630532524673605,0.3851066499699086>, 0.5}

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
    sphere { m*<0.8781810153327281,0.6630532524673605,0.3851066499699086>, 1 }        
    sphere {  m*<1.121631880034474,0.7196744488307629,3.374673823477221>, 1 }
    sphere {  m*<3.6148790690970105,0.7196744488307627,-0.8426083850133965>, 1 }
    sphere {  m*<-2.445355063418057,5.760952913218832,-1.579975969539605>, 1}
    sphere { m*<-3.858763143237336,-7.686285202753245,-2.415009660965361>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.121631880034474,0.7196744488307629,3.374673823477221>, <0.8781810153327281,0.6630532524673605,0.3851066499699086>, 0.5 }
    cylinder { m*<3.6148790690970105,0.7196744488307627,-0.8426083850133965>, <0.8781810153327281,0.6630532524673605,0.3851066499699086>, 0.5}
    cylinder { m*<-2.445355063418057,5.760952913218832,-1.579975969539605>, <0.8781810153327281,0.6630532524673605,0.3851066499699086>, 0.5 }
    cylinder {  m*<-3.858763143237336,-7.686285202753245,-2.415009660965361>, <0.8781810153327281,0.6630532524673605,0.3851066499699086>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    