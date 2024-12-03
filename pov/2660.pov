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
    sphere { m*<0.7638034430018835,0.833052383798564,0.31747969851768487>, 1 }        
    sphere {  m*<1.0066680301182074,0.9076442734874465,3.3066995046465424>, 1 }
    sphere {  m*<3.499915219180742,0.9076442734874463,-0.9105827038440721>, 1 }
    sphere {  m*<-2.0667663545672412,5.082296699987935,-1.3561264707760001>, 1}
    sphere { m*<-3.898267899637594,-7.57298023379478,-2.438369579824525>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0066680301182074,0.9076442734874465,3.3066995046465424>, <0.7638034430018835,0.833052383798564,0.31747969851768487>, 0.5 }
    cylinder { m*<3.499915219180742,0.9076442734874463,-0.9105827038440721>, <0.7638034430018835,0.833052383798564,0.31747969851768487>, 0.5}
    cylinder { m*<-2.0667663545672412,5.082296699987935,-1.3561264707760001>, <0.7638034430018835,0.833052383798564,0.31747969851768487>, 0.5 }
    cylinder {  m*<-3.898267899637594,-7.57298023379478,-2.438369579824525>, <0.7638034430018835,0.833052383798564,0.31747969851768487>, 0.5}

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
    sphere { m*<0.7638034430018835,0.833052383798564,0.31747969851768487>, 1 }        
    sphere {  m*<1.0066680301182074,0.9076442734874465,3.3066995046465424>, 1 }
    sphere {  m*<3.499915219180742,0.9076442734874463,-0.9105827038440721>, 1 }
    sphere {  m*<-2.0667663545672412,5.082296699987935,-1.3561264707760001>, 1}
    sphere { m*<-3.898267899637594,-7.57298023379478,-2.438369579824525>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0066680301182074,0.9076442734874465,3.3066995046465424>, <0.7638034430018835,0.833052383798564,0.31747969851768487>, 0.5 }
    cylinder { m*<3.499915219180742,0.9076442734874463,-0.9105827038440721>, <0.7638034430018835,0.833052383798564,0.31747969851768487>, 0.5}
    cylinder { m*<-2.0667663545672412,5.082296699987935,-1.3561264707760001>, <0.7638034430018835,0.833052383798564,0.31747969851768487>, 0.5 }
    cylinder {  m*<-3.898267899637594,-7.57298023379478,-2.438369579824525>, <0.7638034430018835,0.833052383798564,0.31747969851768487>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    