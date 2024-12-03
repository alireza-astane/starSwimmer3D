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
    sphere { m*<-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 1 }        
    sphere {  m*<0.25099632409744416,0.2847582817582388,8.528567577844552>, 1 }
    sphere {  m*<4.959191760009478,0.04669963718478207,-4.285214546301195>, 1 }
    sphere {  m*<-2.5577725639876165,2.1666156814823876,-2.314477046461417>, 1}
    sphere { m*<-2.289985342949785,-2.72107626092151,-2.124930761298846>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25099632409744416,0.2847582817582388,8.528567577844552>, <-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 0.5 }
    cylinder { m*<4.959191760009478,0.04669963718478207,-4.285214546301195>, <-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 0.5}
    cylinder { m*<-2.5577725639876165,2.1666156814823876,-2.314477046461417>, <-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 0.5 }
    cylinder {  m*<-2.289985342949785,-2.72107626092151,-2.124930761298846>, <-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 0.5}

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
    sphere { m*<-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 1 }        
    sphere {  m*<0.25099632409744416,0.2847582817582388,8.528567577844552>, 1 }
    sphere {  m*<4.959191760009478,0.04669963718478207,-4.285214546301195>, 1 }
    sphere {  m*<-2.5577725639876165,2.1666156814823876,-2.314477046461417>, 1}
    sphere { m*<-2.289985342949785,-2.72107626092151,-2.124930761298846>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25099632409744416,0.2847582817582388,8.528567577844552>, <-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 0.5 }
    cylinder { m*<4.959191760009478,0.04669963718478207,-4.285214546301195>, <-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 0.5}
    cylinder { m*<-2.5577725639876165,2.1666156814823876,-2.314477046461417>, <-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 0.5 }
    cylinder {  m*<-2.289985342949785,-2.72107626092151,-2.124930761298846>, <-0.9056727168695995,-0.1622675849802375,-1.3942265346364808>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    