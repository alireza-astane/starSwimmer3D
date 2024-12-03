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
    sphere { m*<-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 1 }        
    sphere {  m*<0.18378781361211843,0.10304516043557285,3.9251304056538574>, 1 }
    sphere {  m*<2.565979769241625,0.01660467800655749,-1.678855156308184>, 1 }
    sphere {  m*<-1.7903439846575218,2.243044647038782,-1.4235913962729705>, 1}
    sphere { m*<-1.5225567636196902,-2.6446472953651154,-1.2340451111103978>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18378781361211843,0.10304516043557285,3.9251304056538574>, <-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 0.5 }
    cylinder { m*<2.565979769241625,0.01660467800655749,-1.678855156308184>, <-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 0.5}
    cylinder { m*<-1.7903439846575218,2.243044647038782,-1.4235913962729705>, <-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 0.5 }
    cylinder {  m*<-1.5225567636196902,-2.6446472953651154,-1.2340451111103978>, <-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 0.5}

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
    sphere { m*<-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 1 }        
    sphere {  m*<0.18378781361211843,0.10304516043557285,3.9251304056538574>, 1 }
    sphere {  m*<2.565979769241625,0.01660467800655749,-1.678855156308184>, 1 }
    sphere {  m*<-1.7903439846575218,2.243044647038782,-1.4235913962729705>, 1}
    sphere { m*<-1.5225567636196902,-2.6446472953651154,-1.2340451111103978>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.18378781361211843,0.10304516043557285,3.9251304056538574>, <-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 0.5 }
    cylinder { m*<2.565979769241625,0.01660467800655749,-1.678855156308184>, <-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 0.5}
    cylinder { m*<-1.7903439846575218,2.243044647038782,-1.4235913962729705>, <-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 0.5 }
    cylinder {  m*<-1.5225567636196902,-2.6446472953651154,-1.2340451111103978>, <-0.16872862476463174,-0.08542929737981664,-0.4496456308570005>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    