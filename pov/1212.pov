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
    sphere { m*<0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 1 }        
    sphere {  m*<0.33697050930504946,-3.4167523532062173e-18,4.088101505973824>, 1 }
    sphere {  m*<8.279539664348452,4.420610338570081e-18,-1.8564143458038729>, 1 }
    sphere {  m*<-4.4624057182779335,8.164965809277259,-2.1808198461957904>, 1}
    sphere { m*<-4.4624057182779335,-8.164965809277259,-2.180819846195794>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33697050930504946,-3.4167523532062173e-18,4.088101505973824>, <0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 0.5 }
    cylinder { m*<8.279539664348452,4.420610338570081e-18,-1.8564143458038729>, <0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 0.5}
    cylinder { m*<-4.4624057182779335,8.164965809277259,-2.1808198461957904>, <0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 0.5 }
    cylinder {  m*<-4.4624057182779335,-8.164965809277259,-2.180819846195794>, <0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 0.5}

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
    sphere { m*<0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 1 }        
    sphere {  m*<0.33697050930504946,-3.4167523532062173e-18,4.088101505973824>, 1 }
    sphere {  m*<8.279539664348452,4.420610338570081e-18,-1.8564143458038729>, 1 }
    sphere {  m*<-4.4624057182779335,8.164965809277259,-2.1808198461957904>, 1}
    sphere { m*<-4.4624057182779335,-8.164965809277259,-2.180819846195794>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33697050930504946,-3.4167523532062173e-18,4.088101505973824>, <0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 0.5 }
    cylinder { m*<8.279539664348452,4.420610338570081e-18,-1.8564143458038729>, <0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 0.5}
    cylinder { m*<-4.4624057182779335,8.164965809277259,-2.1808198461957904>, <0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 0.5 }
    cylinder {  m*<-4.4624057182779335,-8.164965809277259,-2.180819846195794>, <0.29634678181000457,-4.287069993050261e-18,1.088375272642091>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    