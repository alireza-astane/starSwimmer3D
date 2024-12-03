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
    sphere { m*<0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 1 }        
    sphere {  m*<1.1230800061030264,1.0294461203115424e-18,3.795226757146196>, 1 }
    sphere {  m*<5.539019479309016,5.603475538379161e-18,-1.106092496634482>, 1 }
    sphere {  m*<-3.9288304209363747,8.164965809277259,-2.2720457590360326>, 1}
    sphere { m*<-3.9288304209363747,-8.164965809277259,-2.2720457590360352>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1230800061030264,1.0294461203115424e-18,3.795226757146196>, <0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 0.5 }
    cylinder { m*<5.539019479309016,5.603475538379161e-18,-1.106092496634482>, <0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 0.5}
    cylinder { m*<-3.9288304209363747,8.164965809277259,-2.2720457590360326>, <0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 0.5 }
    cylinder {  m*<-3.9288304209363747,-8.164965809277259,-2.2720457590360352>, <0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 0.5}

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
    sphere { m*<0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 1 }        
    sphere {  m*<1.1230800061030264,1.0294461203115424e-18,3.795226757146196>, 1 }
    sphere {  m*<5.539019479309016,5.603475538379161e-18,-1.106092496634482>, 1 }
    sphere {  m*<-3.9288304209363747,8.164965809277259,-2.2720457590360326>, 1}
    sphere { m*<-3.9288304209363747,-8.164965809277259,-2.2720457590360352>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1230800061030264,1.0294461203115424e-18,3.795226757146196>, <0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 0.5 }
    cylinder { m*<5.539019479309016,5.603475538379161e-18,-1.106092496634482>, <0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 0.5}
    cylinder { m*<-3.9288304209363747,8.164965809277259,-2.2720457590360326>, <0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 0.5 }
    cylinder {  m*<-3.9288304209363747,-8.164965809277259,-2.2720457590360352>, <0.9597306907485207,-1.8798207547913055e-18,0.7996709246500362>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    