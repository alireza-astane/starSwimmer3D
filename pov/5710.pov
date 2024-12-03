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
    sphere { m*<-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 1 }        
    sphere {  m*<0.10585340774800245,0.2817350376400585,8.655659830346911>, 1 }
    sphere {  m*<5.915075596524681,0.07624338409577117,-4.8731374447067>, 1 }
    sphere {  m*<-2.850688861654469,2.156988141190013,-2.1491663498901525>, 1}
    sphere { m*<-2.5829016406166376,-2.7307038012138842,-1.959620064727582>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10585340774800245,0.2817350376400585,8.655659830346911>, <-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 0.5 }
    cylinder { m*<5.915075596524681,0.07624338409577117,-4.8731374447067>, <-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 0.5}
    cylinder { m*<-2.850688861654469,2.156988141190013,-2.1491663498901525>, <-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 0.5 }
    cylinder {  m*<-2.5829016406166376,-2.7307038012138842,-1.959620064727582>, <-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 0.5}

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
    sphere { m*<-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 1 }        
    sphere {  m*<0.10585340774800245,0.2817350376400585,8.655659830346911>, 1 }
    sphere {  m*<5.915075596524681,0.07624338409577117,-4.8731374447067>, 1 }
    sphere {  m*<-2.850688861654469,2.156988141190013,-2.1491663498901525>, 1}
    sphere { m*<-2.5829016406166376,-2.7307038012138842,-1.959620064727582>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10585340774800245,0.2817350376400585,8.655659830346911>, <-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 0.5 }
    cylinder { m*<5.915075596524681,0.07624338409577117,-4.8731374447067>, <-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 0.5}
    cylinder { m*<-2.850688861654469,2.156988141190013,-2.1491663498901525>, <-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 0.5 }
    cylinder {  m*<-2.5829016406166376,-2.7307038012138842,-1.959620064727582>, <-1.187323972742006,-0.17209246721196209,-1.2499496054815729>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    