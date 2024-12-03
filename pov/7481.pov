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
    sphere { m*<-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 1 }        
    sphere {  m*<0.837492498717631,0.20262411799248836,9.3303596429686>, 1 }
    sphere {  m*<8.205279697040432,-0.08246813279977339,-5.240317786105331>, 1 }
    sphere {  m*<-6.690683496648556,6.44061324082086,-3.7495108829237243>, 1}
    sphere { m*<-3.1931752586566335,-6.474652453244441,-1.728283711121671>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.837492498717631,0.20262411799248836,9.3303596429686>, <-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 0.5 }
    cylinder { m*<8.205279697040432,-0.08246813279977339,-5.240317786105331>, <-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 0.5}
    cylinder { m*<-6.690683496648556,6.44061324082086,-3.7495108829237243>, <-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 0.5 }
    cylinder {  m*<-3.1931752586566335,-6.474652453244441,-1.728283711121671>, <-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 0.5}

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
    sphere { m*<-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 1 }        
    sphere {  m*<0.837492498717631,0.20262411799248836,9.3303596429686>, 1 }
    sphere {  m*<8.205279697040432,-0.08246813279977339,-5.240317786105331>, 1 }
    sphere {  m*<-6.690683496648556,6.44061324082086,-3.7495108829237243>, 1}
    sphere { m*<-3.1931752586566335,-6.474652453244441,-1.728283711121671>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.837492498717631,0.20262411799248836,9.3303596429686>, <-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 0.5 }
    cylinder { m*<8.205279697040432,-0.08246813279977339,-5.240317786105331>, <-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 0.5}
    cylinder { m*<-6.690683496648556,6.44061324082086,-3.7495108829237243>, <-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 0.5 }
    cylinder {  m*<-3.1931752586566335,-6.474652453244441,-1.728283711121671>, <-0.5816749954825311,-0.7873147958874289,-0.5189304540665495>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    