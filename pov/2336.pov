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
    sphere { m*<1.0179666368660003,0.443315660719843,0.4677568871009213>, 1 }        
    sphere {  m*<1.261898277363221,0.47897559421063945,3.4576091822985653>, 1 }
    sphere {  m*<3.755145466425757,0.4789755942106393,-0.7596730261920535>, 1 }
    sphere {  m*<-2.8892970946362815,6.5899409891982135,-1.8424685666785672>, 1}
    sphere { m*<-3.805633342112355,-7.838781826417548,-2.383592987484879>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.261898277363221,0.47897559421063945,3.4576091822985653>, <1.0179666368660003,0.443315660719843,0.4677568871009213>, 0.5 }
    cylinder { m*<3.755145466425757,0.4789755942106393,-0.7596730261920535>, <1.0179666368660003,0.443315660719843,0.4677568871009213>, 0.5}
    cylinder { m*<-2.8892970946362815,6.5899409891982135,-1.8424685666785672>, <1.0179666368660003,0.443315660719843,0.4677568871009213>, 0.5 }
    cylinder {  m*<-3.805633342112355,-7.838781826417548,-2.383592987484879>, <1.0179666368660003,0.443315660719843,0.4677568871009213>, 0.5}

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
    sphere { m*<1.0179666368660003,0.443315660719843,0.4677568871009213>, 1 }        
    sphere {  m*<1.261898277363221,0.47897559421063945,3.4576091822985653>, 1 }
    sphere {  m*<3.755145466425757,0.4789755942106393,-0.7596730261920535>, 1 }
    sphere {  m*<-2.8892970946362815,6.5899409891982135,-1.8424685666785672>, 1}
    sphere { m*<-3.805633342112355,-7.838781826417548,-2.383592987484879>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.261898277363221,0.47897559421063945,3.4576091822985653>, <1.0179666368660003,0.443315660719843,0.4677568871009213>, 0.5 }
    cylinder { m*<3.755145466425757,0.4789755942106393,-0.7596730261920535>, <1.0179666368660003,0.443315660719843,0.4677568871009213>, 0.5}
    cylinder { m*<-2.8892970946362815,6.5899409891982135,-1.8424685666785672>, <1.0179666368660003,0.443315660719843,0.4677568871009213>, 0.5 }
    cylinder {  m*<-3.805633342112355,-7.838781826417548,-2.383592987484879>, <1.0179666368660003,0.443315660719843,0.4677568871009213>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    