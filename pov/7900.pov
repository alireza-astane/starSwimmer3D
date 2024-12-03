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
    sphere { m*<-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 1 }        
    sphere {  m*<1.051659912018164,0.6690389262998122,9.429537905549072>, 1 }
    sphere {  m*<8.41944711034096,0.38394667550755046,-5.141139523524852>, 1 }
    sphere {  m*<-6.476516083348035,6.907028049128182,-3.650332620343244>, 1}
    sphere { m*<-4.180691289329992,-8.625269547828255,-2.185590089730815>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.051659912018164,0.6690389262998122,9.429537905549072>, <-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 0.5 }
    cylinder { m*<8.41944711034096,0.38394667550755046,-5.141139523524852>, <-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 0.5}
    cylinder { m*<-6.476516083348035,6.907028049128182,-3.650332620343244>, <-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 0.5 }
    cylinder {  m*<-4.180691289329992,-8.625269547828255,-2.185590089730815>, <-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 0.5}

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
    sphere { m*<-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 1 }        
    sphere {  m*<1.051659912018164,0.6690389262998122,9.429537905549072>, 1 }
    sphere {  m*<8.41944711034096,0.38394667550755046,-5.141139523524852>, 1 }
    sphere {  m*<-6.476516083348035,6.907028049128182,-3.650332620343244>, 1}
    sphere { m*<-4.180691289329992,-8.625269547828255,-2.185590089730815>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.051659912018164,0.6690389262998122,9.429537905549072>, <-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 0.5 }
    cylinder { m*<8.41944711034096,0.38394667550755046,-5.141139523524852>, <-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 0.5}
    cylinder { m*<-6.476516083348035,6.907028049128182,-3.650332620343244>, <-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 0.5 }
    cylinder {  m*<-4.180691289329992,-8.625269547828255,-2.185590089730815>, <-0.3675075821819969,-0.3208999875801045,-0.41975219148606957>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    