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
    sphere { m*<-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 1 }        
    sphere {  m*<0.8683220258709562,0.26976480842631156,9.344636413146299>, 1 }
    sphere {  m*<8.236109224193763,-0.015327442365949517,-5.22604101592763>, 1 }
    sphere {  m*<-6.659853969495235,6.507753931254691,-3.7352341127460233>, 1}
    sphere { m*<-3.3401604654564845,-6.794757533644154,-1.7963507303324102>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8683220258709562,0.26976480842631156,9.344636413146299>, <-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 0.5 }
    cylinder { m*<8.236109224193763,-0.015327442365949517,-5.22604101592763>, <-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 0.5}
    cylinder { m*<-6.659853969495235,6.507753931254691,-3.7352341127460233>, <-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 0.5 }
    cylinder {  m*<-3.3401604654564845,-6.794757533644154,-1.7963507303324102>, <-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 0.5}

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
    sphere { m*<-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 1 }        
    sphere {  m*<0.8683220258709562,0.26976480842631156,9.344636413146299>, 1 }
    sphere {  m*<8.236109224193763,-0.015327442365949517,-5.22604101592763>, 1 }
    sphere {  m*<-6.659853969495235,6.507753931254691,-3.7352341127460233>, 1}
    sphere { m*<-3.3401604654564845,-6.794757533644154,-1.7963507303324102>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8683220258709562,0.26976480842631156,9.344636413146299>, <-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 0.5 }
    cylinder { m*<8.236109224193763,-0.015327442365949517,-5.22604101592763>, <-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 0.5}
    cylinder { m*<-6.659853969495235,6.507753931254691,-3.7352341127460233>, <-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 0.5 }
    cylinder {  m*<-3.3401604654564845,-6.794757533644154,-1.7963507303324102>, <-0.5508454683292053,-0.7201741054536056,-0.5046536838888478>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    