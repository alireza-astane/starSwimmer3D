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
    sphere { m*<-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 1 }        
    sphere {  m*<0.981802847208063,0.5169038792812715,9.397187968599914>, 1 }
    sphere {  m*<8.34959004553086,0.23181162848900905,-5.173489460474018>, 1 }
    sphere {  m*<-6.546373148158134,6.754893002109648,-3.682682557292411>, 1}
    sphere { m*<-3.866239090434839,-7.940454057883997,-2.0399711915501255>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.981802847208063,0.5169038792812715,9.397187968599914>, <-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 0.5 }
    cylinder { m*<8.34959004553086,0.23181162848900905,-5.173489460474018>, <-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 0.5}
    cylinder { m*<-6.546373148158134,6.754893002109648,-3.682682557292411>, <-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 0.5 }
    cylinder {  m*<-3.866239090434839,-7.940454057883997,-2.0399711915501255>, <-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 0.5}

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
    sphere { m*<-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 1 }        
    sphere {  m*<0.981802847208063,0.5169038792812715,9.397187968599914>, 1 }
    sphere {  m*<8.34959004553086,0.23181162848900905,-5.173489460474018>, 1 }
    sphere {  m*<-6.546373148158134,6.754893002109648,-3.682682557292411>, 1}
    sphere { m*<-3.866239090434839,-7.940454057883997,-2.0399711915501255>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.981802847208063,0.5169038792812715,9.397187968599914>, <-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 0.5 }
    cylinder { m*<8.34959004553086,0.23181162848900905,-5.173489460474018>, <-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 0.5}
    cylinder { m*<-6.546373148158134,6.754893002109648,-3.682682557292411>, <-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 0.5 }
    cylinder {  m*<-3.866239090434839,-7.940454057883997,-2.0399711915501255>, <-0.4373646469920987,-0.4730350345986458,-0.452102128435234>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    