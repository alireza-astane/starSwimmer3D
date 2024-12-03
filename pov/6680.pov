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
    sphere { m*<-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 1 }        
    sphere {  m*<0.3474621726548044,-0.15258321380946205,9.097726831722508>, 1 }
    sphere {  m*<7.702813610654776,-0.24150348980381842,-5.481766458322831>, 1 }
    sphere {  m*<-5.764781782993847,4.792855958170524,-3.1617399146243357>, 1}
    sphere { m*<-2.3524806803507787,-3.5781886753806504,-1.3890048734745546>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3474621726548044,-0.15258321380946205,9.097726831722508>, <-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 0.5 }
    cylinder { m*<7.702813610654776,-0.24150348980381842,-5.481766458322831>, <-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 0.5}
    cylinder { m*<-5.764781782993847,4.792855958170524,-3.1617399146243357>, <-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 0.5 }
    cylinder {  m*<-2.3524806803507787,-3.5781886753806504,-1.3890048734745546>, <-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 0.5}

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
    sphere { m*<-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 1 }        
    sphere {  m*<0.3474621726548044,-0.15258321380946205,9.097726831722508>, 1 }
    sphere {  m*<7.702813610654776,-0.24150348980381842,-5.481766458322831>, 1 }
    sphere {  m*<-5.764781782993847,4.792855958170524,-3.1617399146243357>, 1}
    sphere { m*<-2.3524806803507787,-3.5781886753806504,-1.3890048734745546>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3474621726548044,-0.15258321380946205,9.097726831722508>, <-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 0.5 }
    cylinder { m*<7.702813610654776,-0.24150348980381842,-5.481766458322831>, <-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 0.5}
    cylinder { m*<-5.764781782993847,4.792855958170524,-3.1617399146243357>, <-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 0.5 }
    cylinder {  m*<-2.3524806803507787,-3.5781886753806504,-1.3890048734745546>, <-1.089598971230635,-0.9290616529061363,-0.7680423076159443>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    