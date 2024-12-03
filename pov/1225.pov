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
    sphere { m*<0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 1 }        
    sphere {  m*<0.3574765864004347,-3.1316304315507577e-18,4.08127491353292>, 1 }
    sphere {  m*<8.209387473526046,4.238146038706099e-18,-1.8387629050854313>, 1 }
    sphere {  m*<-4.447492058489505,8.164965809277259,-2.183361327581234>, 1}
    sphere { m*<-4.447492058489505,-8.164965809277259,-2.1833613275812374>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3574765864004347,-3.1316304315507577e-18,4.08127491353292>, <0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 0.5 }
    cylinder { m*<8.209387473526046,4.238146038706099e-18,-1.8387629050854313>, <0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 0.5}
    cylinder { m*<-4.447492058489505,8.164965809277259,-2.183361327581234>, <0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 0.5 }
    cylinder {  m*<-4.447492058489505,-8.164965809277259,-2.1833613275812374>, <0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 0.5}

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
    sphere { m*<0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 1 }        
    sphere {  m*<0.3574765864004347,-3.1316304315507577e-18,4.08127491353292>, 1 }
    sphere {  m*<8.209387473526046,4.238146038706099e-18,-1.8387629050854313>, 1 }
    sphere {  m*<-4.447492058489505,8.164965809277259,-2.183361327581234>, 1}
    sphere { m*<-4.447492058489505,-8.164965809277259,-2.1833613275812374>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3574765864004347,-3.1316304315507577e-18,4.08127491353292>, <0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 0.5 }
    cylinder { m*<8.209387473526046,4.238146038706099e-18,-1.8387629050854313>, <0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 0.5}
    cylinder { m*<-4.447492058489505,8.164965809277259,-2.183361327581234>, <0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 0.5 }
    cylinder {  m*<-4.447492058489505,-8.164965809277259,-2.1833613275812374>, <0.31417601584857824,-4.4875218506651765e-18,1.0815860339878336>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    