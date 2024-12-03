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
    sphere { m*<-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 1 }        
    sphere {  m*<0.027478556428423673,0.11323576935627877,8.934663742020916>, 1 }
    sphere {  m*<7.382829994428396,0.024315493361921625,-5.644829548024438>, 1 }
    sphere {  m*<-4.177622762883373,3.1625336215718325,-2.3510209038987>, 1}
    sphere { m*<-2.7768023761106733,-3.05228603955345,-1.6062981482449956>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.027478556428423673,0.11323576935627877,8.934663742020916>, <-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 0.5 }
    cylinder { m*<7.382829994428396,0.024315493361921625,-5.644829548024438>, <-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 0.5}
    cylinder { m*<-4.177622762883373,3.1625336215718325,-2.3510209038987>, <-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 0.5 }
    cylinder {  m*<-2.7768023761106733,-3.05228603955345,-1.6062981482449956>, <-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 0.5}

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
    sphere { m*<-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 1 }        
    sphere {  m*<0.027478556428423673,0.11323576935627877,8.934663742020916>, 1 }
    sphere {  m*<7.382829994428396,0.024315493361921625,-5.644829548024438>, 1 }
    sphere {  m*<-4.177622762883373,3.1625336215718325,-2.3510209038987>, 1}
    sphere { m*<-2.7768023761106733,-3.05228603955345,-1.6062981482449956>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.027478556428423673,0.11323576935627877,8.934663742020916>, <-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 0.5 }
    cylinder { m*<7.382829994428396,0.024315493361921625,-5.644829548024438>, <-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 0.5}
    cylinder { m*<-4.177622762883373,3.1625336215718325,-2.3510209038987>, <-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 0.5 }
    cylinder {  m*<-2.7768023761106733,-3.05228603955345,-1.6062981482449956>, <-1.4293673787984824,-0.455766938391847,-0.9423551279408567>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    