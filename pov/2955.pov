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
    sphere { m*<0.5423154668176944,1.1365569085485363,0.18652367463414277>, 1 }        
    sphere {  m*<0.783512138303514,1.2473132624125132,3.1747558806307135>, 1 }
    sphere {  m*<3.2767593273660482,1.2473132624125127,-1.0425263278599022>, 1 }
    sphere {  m*<-1.247171154440665,3.709735679047252,-0.8715255829377142>, 1}
    sphere { m*<-3.967453947489066,-7.380894750777026,-2.4792803455852424>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.783512138303514,1.2473132624125132,3.1747558806307135>, <0.5423154668176944,1.1365569085485363,0.18652367463414277>, 0.5 }
    cylinder { m*<3.2767593273660482,1.2473132624125127,-1.0425263278599022>, <0.5423154668176944,1.1365569085485363,0.18652367463414277>, 0.5}
    cylinder { m*<-1.247171154440665,3.709735679047252,-0.8715255829377142>, <0.5423154668176944,1.1365569085485363,0.18652367463414277>, 0.5 }
    cylinder {  m*<-3.967453947489066,-7.380894750777026,-2.4792803455852424>, <0.5423154668176944,1.1365569085485363,0.18652367463414277>, 0.5}

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
    sphere { m*<0.5423154668176944,1.1365569085485363,0.18652367463414277>, 1 }        
    sphere {  m*<0.783512138303514,1.2473132624125132,3.1747558806307135>, 1 }
    sphere {  m*<3.2767593273660482,1.2473132624125127,-1.0425263278599022>, 1 }
    sphere {  m*<-1.247171154440665,3.709735679047252,-0.8715255829377142>, 1}
    sphere { m*<-3.967453947489066,-7.380894750777026,-2.4792803455852424>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.783512138303514,1.2473132624125132,3.1747558806307135>, <0.5423154668176944,1.1365569085485363,0.18652367463414277>, 0.5 }
    cylinder { m*<3.2767593273660482,1.2473132624125127,-1.0425263278599022>, <0.5423154668176944,1.1365569085485363,0.18652367463414277>, 0.5}
    cylinder { m*<-1.247171154440665,3.709735679047252,-0.8715255829377142>, <0.5423154668176944,1.1365569085485363,0.18652367463414277>, 0.5 }
    cylinder {  m*<-3.967453947489066,-7.380894750777026,-2.4792803455852424>, <0.5423154668176944,1.1365569085485363,0.18652367463414277>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    